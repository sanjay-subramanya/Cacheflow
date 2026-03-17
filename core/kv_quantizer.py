import torch

class KVQuantizer:
    def quantize(self, x):
        """Quantizer for KV cache compression to int8"""
        # x comprised of [batch, heads, seq, head_dim]
        if x.dim() == 4: 
            scales = x.abs().max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            scales = torch.clamp(scales / 127, min=1e-6)
            q = (x / scales).round().clamp(-128, 127).to(torch.int8)
            return q, scales
        else:
            scale = x.abs().max() / 127
            q = (x / scale).round().clamp(-128, 127).to(torch.int8)
            return q, scale

    def dequantize(self, q, scale):
        """Dequantize int8 KV cache tensor to float"""
        return q.float() * scale
    