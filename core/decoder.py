import gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from .kv_manager import KVMemoryManager
from .telemetry import KVTelemetry

class KVCompressedDecoder:
    """Decoder engine with compressed KV cache for efficient memory management"""
    def __init__(self, config):
        self.last_tokens = [] 
        self.cfg = config
        self.soft_limit = int(0.75 * self.cfg.max_new_tokens)
        self.device = self.cfg.device
        
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            dtype=self.cfg.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.cache_dtype = next(self.model.parameters()).dtype
        self.num_layers = self.model.config.num_hidden_layers
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.global_pos = 0

        self.telemetry = KVTelemetry(
            num_layers=self.num_layers, 
            num_kv_heads=num_kv_heads, 
            head_dim=head_dim
        )
        
        self.max_tokens = self.cfg.max_new_tokens 
        self.kv = KVMemoryManager(
            fp_window=self.cfg.fp16_window,
            int8_window=self.cfg.int8_window,
            cache_dtype=self.cache_dtype
        )
        self.kv.set_num_layers(self.num_layers)
        self.kv.dtype = self.cache_dtype
    
    def reset_cache(self):
        """Reset KV cache to empty state"""     
        if hasattr(self, 'kv'):
            del self.kv
            gc.collect()
        self.global_pos = 0
        self.kv = KVMemoryManager(
            fp_window=self.cfg.fp16_window,
            int8_window=self.cfg.int8_window,
            cache_dtype=self.cache_dtype
        )
        self.kv.dtype = next(self.model.parameters()).dtype
        self.kv.set_num_layers(self.num_layers)
        self.telemetry.reset()

    def generate(self, prompt_chunk):
        """Generate response for a single prompt chunk using compressed KV cache"""
        with torch.no_grad():
            input_ids = self.tok(prompt_chunk, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            chunk_len = input_ids.shape[1]
            
            past = DynamicCache()
            model_dtype = next(self.model.parameters()).dtype        
        
            for layer_idx in range(self.num_layers):
                k, v = self.kv.get_layer_kv(layer_idx, self.device)
                if k is not None and v is not None:
                    k = k.to(dtype=model_dtype, device=self.device)
                    v = v.to(dtype=model_dtype, device=self.device)
                    past.update(k, v, layer_idx)

            position_ids = torch.arange(self.global_pos, self.global_pos + chunk_len, device=self.device).unsqueeze(0)

            out = self.model(
                input_ids=input_ids,
                past_key_values=past if len(past) > 0 else None,
                position_ids=position_ids,
                use_cache=True
            ) 
            new_past = out.past_key_values

            for t in range(chunk_len):
                for layer_idx in range(self.num_layers):

                    if hasattr(new_past, 'layers'):
                        k = new_past.layers[layer_idx].keys
                        v = new_past.layers[layer_idx].values
                    elif hasattr(new_past, 'key_cache'):
                        k = new_past.key_cache[layer_idx]
                        v = new_past.value_cache[layer_idx]
                    elif isinstance(new_past, (tuple, list)):
                        k, v = new_past[layer_idx]
                    else:
                        raise TypeError(f"Unexpected past_key_values type: {type(new_past)}")

                    idx = (k.shape[2] - chunk_len) + t
                    self.kv.append(
                        k[:, :, idx:idx+1, :].to(torch.float32).detach().cpu(), 
                        v[:, :, idx:idx+1, :].to(torch.float32).detach().cpu(), 
                        layer_idx
                    )

            self.global_pos += chunk_len

            logits = out.logits[:, -1, :]
            next_token = self._sample(logits)
            first_id = next_token.item()
            current_response = self.tok.decode([first_id], skip_special_tokens=True)
            self.telemetry.lifetime_tokens += chunk_len
            yield current_response, self.telemetry

            current_past = new_past
            
            for step in range(self.cfg.max_new_tokens):
                position_ids = torch.tensor([[self.global_pos]], device=self.device)

                out = self.model(
                    input_ids=next_token,
                    past_key_values=current_past,
                    position_ids=position_ids,
                    use_cache=True
                )
                
                self.global_pos += 1
                logits = out.logits[:, -1, :].clone()

                if len(self.last_tokens) > 0:
                    for token_id in set(self.last_tokens[-5:]):
                        logits[:, token_id] -= 2.0

                token = self._sample(logits)
                self.last_tokens.append(token.item())
                if len(self.last_tokens) > 20:
                    self.last_tokens.pop(0)
                
                current_past = out.past_key_values
                
                for layer_idx in range(self.num_layers):

                    if hasattr(current_past, 'layers'):
                        k = current_past.layers[layer_idx].keys
                        v = current_past.layers[layer_idx].values
                    elif hasattr(current_past, 'key_cache'):
                        k = current_past.key_cache[layer_idx]
                        v = current_past.value_cache[layer_idx]
                    elif isinstance(current_past, (tuple, list)):
                        k, v = current_past[layer_idx]
                    else:
                        raise TypeError(f"Unexpected past_key_values type: {type(current_past)}")

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
                        break

                if token_id == self.tok.eos_token_id:
                    break

                current_response += text
                yield current_response, self.telemetry

                next_token = token.to(self.device)

                if step % 10 == 0:
                    self.telemetry.update(self.kv)
            
            self.telemetry.update(self.kv)

    def _sample(self, logits, temperature=0.7, top_k=30):
        """Select next token from logits with temperature and top-k sampling"""
        logits = logits / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
