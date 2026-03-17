import torch

class ChatEngine:
    """Manage chat history, formatting and KV cache"""
    def __init__(self, decoder):
        self.decoder = decoder
        self.messages = []
        self.system_prompt = "<|im_start|>system\nYou are Qwen, a Large Language Model. You are a helpful assistant.<|im_end|>\n"
        self._capture_system_kv()

    def clear_history(self):
        """Clear chat history and reset memory manager"""
        self.messages = []
        self.decoder.reset_cache()
        self._capture_system_kv()

    def respond(self, msg):
        """Generate response for a new user message"""
        self.decoder.last_tokens = []
        self.messages.append({"role": "user", "content": msg})
        prompt = self._format_new_message(msg)
        final_answer = ""

        for response_text, telemetry in self.decoder.generate(prompt):
            final_answer = response_text
            yield response_text, telemetry

        self.messages.append({"role": "assistant", "content": final_answer})

    def _format_new_message(self, msg):
        """Format new message according to Qwen's requirements"""
        if len(self.messages) == 1:
            return f"{self.system_prompt}<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        return f"\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"

    def _capture_system_kv(self):
        """Capture system prompt KV cache for efficient memory management"""
        inputs = self.decoder.tok(self.system_prompt, return_tensors="pt").to(self.decoder.device)
        
        with torch.no_grad():
            out = self.decoder.model(
                input_ids=inputs.input_ids,
                use_cache=True
            )

            all_k = []
            all_v = []
            pkv = out.past_key_values
            
            for layer_idx in range(self.decoder.num_layers):

                if hasattr(pkv, 'layers'):
                    layer = pkv.layers[layer_idx]
                    k = layer.keys
                    v = layer.values
                elif hasattr(pkv, 'key_cache'):
                    k = pkv.key_cache[layer_idx]
                    v = pkv.value_cache[layer_idx]
                elif isinstance(pkv, (tuple, list)):
                    k, v = pkv[layer_idx]
                else:
                    print(type(pkv))
                    print(dir(pkv))
                    raise TypeError(f"Unexpected past_key_values type: {type(pkv)}")

                all_k.append(k.cpu())
                all_v.append(v.cpu())
            
            system_k = torch.stack(all_k)
            system_v = torch.stack(all_v)
            self.decoder.kv.set_system_prompt(system_k, system_v)
