import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from .kv_manager import KVMemoryManager
from .telemetry import KVTelemetry

class KVCompressedDecoder:
    def __init__(self, config):
        self._prefilled_tokens = 0 
        self.last_tokens = [] 
        self.cfg = config
        self._last_input_length = 0
        self._kv_seen_tokens = 0
        
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        if self.device == "cpu":
            torch.set_num_threads(2)  # Match vCPU count exactly

        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            # torch_dtype=torch.float16,
            dtype=torch.bfloat16,
            # dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.num_layers = self.model.config.num_hidden_layers
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        self.telemetry = KVTelemetry(
            num_layers=self.num_layers, 
            num_kv_heads=num_kv_heads, 
            head_dim=head_dim
        )
        
        self.max_tokens = self.cfg.max_new_tokens
        print("num layers ", self.num_layers)     
        self.kv = KVMemoryManager(
            self.cfg.fp16_window,
            self.cfg.int8_window
        )
        self.kv.set_num_layers(self.num_layers)
        self.kv.dtype = next(self.model.parameters()).dtype
    
    def reset_cache(self):
        """Reset KV cache to empty state"""        
        if hasattr(self, 'kv'):
            del self.kv
            gc.collect()
        self._prefilled_tokens = 0 
        self.kv = KVMemoryManager(
            self.cfg.fp16_window,
            self.cfg.int8_window,
        )
        self._last_input_length = 0
        self._kv_seen_tokens = 0
        self.kv.set_num_layers(self.num_layers)
        self.telemetry.reset()

    def generate_final(self, prompt_chunk):
        with torch.no_grad():
            print(f"🔍 RAW PROMPT CHUNK: {repr(prompt_chunk)}")
            input_ids = self.tok(prompt_chunk, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            chunk_len = input_ids.shape[1]
            
            # Build past from compressed cache
            past = DynamicCache()
            model_dtype = next(self.model.parameters()).dtype
            
            history_len = self.kv.fp_token_count + self.kv.int8_token_count
            print(f"🔍 Stateful Start: History={history_len}, New Chunk={chunk_len}")
            
            for layer_idx in range(self.num_layers):
                k, v = self.kv.get_layer_kv(layer_idx, self.device)
                if k is not None and v is not None:
                    k = k.to(dtype=model_dtype, device=self.device)
                    v = v.to(dtype=model_dtype, device=self.device)
                    past.update(k, v, layer_idx)
            
            current_cache_len = past[0][0].shape[2] if len(past) > 0 else 0
            print(f"🔍 INTERNAL CHECK: Model is receiving {current_cache_len} tokens of history.")
            
            # First, process the prompt (no streaming for prompt itself)
            out = self.model(
                input_ids=input_ids,
                past_key_values=past if len(past) > 0 else None,
                use_cache=True
            )
            print(f"🔍 Prompt logits shape: {out.logits.shape}")
            actual_len = out.past_key_values[0][0].shape[2]
            print(f"🔍 Model Position: {actual_len} (Expected {history_len + chunk_len})")
            
            new_past = out.past_key_values

            for t in range(chunk_len):
                for layer_idx in range(self.num_layers):
                    k, v = new_past[layer_idx]
                    # We only take the tokens that were JUST added (the tail of the cache)
                    # Correct index calculation is: (total_len - chunk_len) + t
                    idx = (k.shape[2] - chunk_len) + t
                    self.kv.append(
                        k[:, :, idx:idx+1, :].to(torch.float32).detach().cpu(), 
                        v[:, :, idx:idx+1, :].to(torch.float32).detach().cpu(), 
                        layer_idx
                    )

            self.telemetry.lifetime_tokens += chunk_len
            logits = out.logits[:, -1, :]
            next_token = self._sample(logits)
            first_id = next_token.item()
            current_response = self.tok.decode([first_id], skip_special_tokens=True)
            yield current_response, self.telemetry

            current_past = new_past
            
            for step in range(self.cfg.max_new_tokens):

                out = self.model(
                    input_ids=next_token,
                    past_key_values=current_past,
                    use_cache=True
                )
                logits = out.logits[:, -1, :].clone()
                logits = logits / 0.7

                if len(self.last_tokens) > 0:
                    for token_id in set(self.last_tokens[-5:]):
                        logits[:, token_id] -= 1.5

                token = self._sample(logits)
                self.last_tokens.append(token.item())
                if len(self.last_tokens) > 20:
                    self.last_tokens.pop(0)
                
                current_past = out.past_key_values
                
                # Store token in KV cache
                for layer_idx in range(self.num_layers):
                    k, v = current_past[layer_idx]
                    new_k = k[:, :, -1:, :].to(torch.float32).detach().cpu()
                    new_v = v[:, :, -1:, :].to(torch.float32).detach().cpu()
                    self.kv.append(new_k, new_v, layer_idx)
                
                self.telemetry.lifetime_tokens += 1
                token_id = token.item()
                text = self.tok.decode([token_id], skip_special_tokens=True)

                if step > self.soft_limit:
                    if text.strip() in ['.', '!', '?']:
                        current_response += text
                        yield current_response, self.telemetry
                        print(f"🔍 Natural break reached at step {step}")
                        break

                if token_id == self.tok.eos_token_id:
                    print(f"🔍 EOS reached at step {step}")
                    break

                current_response += text
                yield current_response, self.telemetry

                next_token = token.to(self.device)

                if step % 10 == 0:
                    self.telemetry.update(self.kv)
            
            self.telemetry.update(self.kv)

    def generate(self, prompt):
        # print("max new tokens: ", self.cfg.max_new_tokens)
        # print(f"fp16 window: {self.cfg.fp16_window}, int8 window: {self.cfg.int8_window}")
        with torch.no_grad():
            print(f"🔍 RAW PROMPT: {repr(prompt)}")
            # Tokenize input
            input_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Build past from compressed cache
            past = DynamicCache()
            model_dtype = next(self.model.parameters()).dtype
            
            for layer_idx in range(self.num_layers):
                k, v = self.kv.get_layer_kv(layer_idx, self.device)
                if k is not None and v is not None:
                    k = k.to(dtype=model_dtype, device=self.device)
                    v = v.to(dtype=model_dtype, device=self.device)
                    past.update(k, v, layer_idx)

            # DEBUG 1: After building past
            print(f"🔍 Past built: {len(past)} layers")
            if len(past) > 0:
                k0, _ = past[0]
                print(f"🔍 Cache shape: {k0.shape}, tokens: {k0.shape[2]}")
            
            # DEBUG 2: Check for cache corruption
            for i in range(min(3, len(past))):
                k, v = past[i]
                if torch.isnan(k).any() or torch.isinf(k).any():
                    print(f"❌ NaN/Inf in layer {i} keys!")
            
            # First, process the prompt (no streaming for prompt itself)
            out = self.model(
                input_ids=input_ids,
                past_key_values=past if len(past) > 0 else None,
                use_cache=True
            )
            print(f"🔍 Prompt logits shape: {out.logits.shape}")
            
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1:, :], dim=-1)

            # DEBUG 4: First token
            print(f"🔍 First token ID: {next_token.item()}")
            print(f"🔍 First token text: '{self.tok.decode([next_token.item()])}'")
            
            # Store prompt tokens in KV cache
            prompt_len = input_ids.shape[1]
            for t in range(prompt_len):
                for layer_idx in range(self.num_layers):
                    k, v = past[layer_idx]
                    new_k = k[:, :, t:t+1, :].to(torch.float32).detach().cpu()
                    new_v = v[:, :, t:t+1, :].to(torch.float32).detach().cpu()
                    self.kv.append(new_k, new_v, layer_idx)
            
            self.telemetry.lifetime_tokens += prompt_len
            
            # Streaming generation loop
            first_id = next_token.item()
            current_response = self.tok.decode([first_id], skip_special_tokens=True)
            yield current_response, self.telemetry

            generated_ids = []
            
            for step in range(self.cfg.max_new_tokens):
                # DEBUG 5: Current state
                print(f"\n🔍 Step {step} starting. Input shape: {next_token.shape}")
                step_start = time.time()

                out = self.model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True
                )
                
                # Get next token
                logits = out.logits[:, -1, :].clone()
                logits = logits / 0.7

                if len(self.last_tokens) > 0:
                    for token_id in set(self.last_tokens[-5:]):
                        logits[:, token_id] -= 2.0

                # 3. Convert to probabilities and sample
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)

                # Track tokens
                self.last_tokens.append(token.item())
                if len(self.last_tokens) > 20:
                    self.last_tokens.pop(0)
                # token = torch.argmax(logits, dim=-1)

                # DEBUG 6: Logits analysis
                probs = torch.softmax(logits, dim=-1)
                top5_probs, top5_indices = torch.topk(probs, 5)
                print(f"🔍 Top 5 tokens:")
                for i in range(5):
                    tid = top5_indices[0, i].item()
                    ttxt = self.tok.decode([tid]).replace('\n', '\\n')
                    print(f"   {i+1}. '{ttxt}' (prob: {top5_probs[0, i].item():.4f})")
                
                # Update past
                past = out.past_key_values
                
                # Store token in KV cache
                for layer_idx in range(self.num_layers):
                    k, v = past[layer_idx]
                    new_k = k[:, :, -1:, :].to(torch.float32).detach().cpu()
                    new_v = v[:, :, -1:, :].to(torch.float32).detach().cpu()
                    self.kv.append(new_k, new_v, layer_idx)
                
                self.telemetry.lifetime_tokens += 1
                
                # Decode token
                token_id = token.item()
                print(f"🔍 Token Shape: {token.shape}")

                # DEBUG 7: Chosen token
                text = self.tok.decode([token_id], skip_special_tokens=True)
                print(f"🔍 Chosen: '{text}' (ID: {token_id})")

                # DEBUG 8: Cache stats
                if step % 10 == 0:
                    print(f"🔍 KV cache: FP16={self.kv.fp_token_count}, INT8={self.kv.int8_token_count}")

                
                if token_id == self.tok.eos_token_id:
                    print(f"🔍 EOS reached at step {step}")
                    break

                # generated_text.append(text)
                current_response += text
                
                # Yield immediately for streaming
                yield current_response, self.telemetry
                # next_token = token.unsqueeze(1)
                generated_ids.append(token_id)
                next_token = token.to(self.device) #torch.tensor([generated_ids]).to(self.device)
                
                # DEBUG 9: Step timing
                print(f"🔍 Step {step} took {time.time() - step_start:.3f}s")

                # Update telemetry periodically
                if step % 10 == 0:
                    self.telemetry.update(self.kv)
            
            # Final telemetry update
            self.telemetry.update(self.kv)