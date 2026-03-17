import torch
from .kv_quantizer import KVQuantizer

class KVMemoryManager:
    """Manage compressed KV cache with FP16 and INT8 quantization"""
    def __init__(self, fp_window=128, int8_window=512, cache_dtype=torch.float16):
        self.fp_window = fp_window
        self.int8_window = int8_window
        self.dtype = cache_dtype
        self.fp_tokens = []
        self.int8_tokens = []
        self.int8_scales = []
        self.num_layers = None
        self._initialized = False
        self.quant = KVQuantizer()
        self.system_kv = None

    def set_num_layers(self, num_layers):
        """Set number of layers in the model, must be called before running inference"""
        self.num_layers = num_layers
        self._initialized = True
        
    def set_system_prompt(self, k, v):
        """Set system prompt KV cache, which is not quantized"""
        self.system_kv = (k.cpu(), v.cpu())

    def append(self, k, v, layer_idx):
        """Append KV cache for a single layer at the end of the sequence"""
        if not self._initialized:
            raise RuntimeError("Must call set_num_layers first")

        if layer_idx == 0:
            self.fp_tokens.append({
                "keys": [None] * self.num_layers,
                "values": [None] * self.num_layers
            })

        token = self.fp_tokens[-1]
        token["keys"][layer_idx] = k[:, :, -1:, :].cpu()
        token["values"][layer_idx] = v[:, :, -1:, :].cpu()

        if layer_idx == self.num_layers - 1:
            while len(self.fp_tokens) > self.fp_window:
                old_token = self.fp_tokens.pop(0)
                q_layers = []
                s_layers = []

                for i in range(self.num_layers):
                    qk, ks = self.quant.quantize(old_token["keys"][i])
                    qv, vs = self.quant.quantize(old_token["values"][i])
                    q_layers.append((qk, qv))
                    s_layers.append((ks, vs))
                
                self.int8_tokens.append(q_layers)
                self.int8_scales.append(s_layers)

            while len(self.int8_tokens) > self.int8_window:
                self.int8_tokens.pop(0)
                self.int8_scales.pop(0)


    def get_layer_kv(self, layer_idx, device):
        """Get KV cache for a specific layer, including system prompt"""
        all_k = []
        all_v = []

        if self.system_kv is not None:
            all_k.append(self.system_kv[0][layer_idx].to(device))
            all_v.append(self.system_kv[1][layer_idx].to(device))

        for token, scales in zip(self.int8_tokens, self.int8_scales):
            qk, qv = token[layer_idx]
            ks, vs = scales[layer_idx]
            all_k.append(self.quant.dequantize(qk, ks).to(device))
            all_v.append(self.quant.dequantize(qv, vs).to(device))

        for token in self.fp_tokens:
            all_k.append(token["keys"][layer_idx].to(device))
            all_v.append(token["values"][layer_idx].to(device))

        if not all_k:
            return None, None
        
        k = torch.cat(all_k, dim=2).to(dtype=self.dtype)
        v = torch.cat(all_v, dim=2).to(dtype=self.dtype)

        return k, v


    @property
    def fp_token_count(self):
        return len(self.fp_tokens)

    @property
    def int8_token_count(self):
        return len(self.int8_tokens)

    @property
    def total_tokens(self):
        return self.fp_token_count + self.int8_token_count

    @property
    def fp_memory_bytes(self):
        total = 0
        for token in self.fp_tokens:
            for i in range(self.num_layers):
                k = token["keys"][i]
                v = token["values"][i]
                if k is None:
                    continue
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total

    @property
    def int8_memory_bytes(self):
        total = 0
        for token in self.int8_tokens:
            for qk, qv in token:
                total += qk.numel() * qk.element_size()
                total += qv.numel() * qv.element_size()
        return total
    