class KVTelemetry:
    """Tracks real-time memory savings and token counts"""
    def __init__(self, num_layers=28, num_kv_heads=2, head_dim=128):
        self.lifetime_tokens = 0
        self.total_tokens = 0
        self.fp_tokens = 0
        self.int8_tokens = 0
        self.memory_fp_mb = 0.0
        self.memory_int8_mb = 0.0
        self.bytes_per_token_fp16 = 0.0
        
    def reset(self):
        """Reset telemetry counters"""
        self.lifetime_tokens = 0
        self.total_tokens = 0
        self.fp_tokens = 0
        self.int8_tokens = 0
        self.memory_fp_mb = 0.0
        self.memory_int8_mb = 0.0
    
    def update(self, kv):
        """Called every step to update and stream telemetry metrics"""
        self.fp_tokens = kv.fp_token_count
        self.int8_tokens = kv.int8_token_count
        self.total_tokens = self.fp_tokens + self.int8_tokens
        self.memory_fp_mb = kv.fp_memory_bytes / (1024 * 1024)
        self.memory_int8_mb = kv.int8_memory_bytes / (1024 * 1024)

        if self.fp_tokens > 0 and self.bytes_per_token_fp16 == 0:
            self.bytes_per_token_fp16 = kv.fp_memory_bytes / self.fp_tokens
    
    @property
    def saved_mb(self):
        if self.total_tokens is None:
            return 0
        
        baseline_mb = (self.lifetime_tokens * self.bytes_per_token_fp16) / (1024 * 1024)
        actual_mb = self.memory_fp_mb + self.memory_int8_mb
        return max(0.0, baseline_mb - actual_mb)
    
    def __str__(self):
        return (f"KV Cache Telemetry\n"
                f"FP16 tokens: {self.fp_tokens}\n"
                f"INT8 tokens: {self.int8_tokens}\n"
                f"Total tokens: {self.lifetime_tokens}\n"
                f"Memory saved vs FP16 cache: {self.saved_mb:.2f} MB")
