import torch
import gc
from core.decoder import KVCompressedDecoder
from core.chat import ChatEngine

class ModelManager:
    """Manage model loading, reloading, and parameter updates"""
    def __init__(self, config):
        self.config = config
        self.decoder = None
        self.chat_engine = None
        self.load_model()

    def load_model(self):
        """Initial load or reload of the model"""
        if self.decoder is not None:
            del self.chat_engine, self.decoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        self.decoder = KVCompressedDecoder(self.config)
        self.chat_engine = ChatEngine(self.decoder)

    def update_and_reload(self, fp16_window, int8_window, max_new_tokens):
        """The function called by Gradio UI when parameters are reset"""
        try:
            self.config.fp16_window = fp16_window
            self.config.int8_window = int8_window
            self.config.max_new_tokens = max_new_tokens
            self.load_model()                
            return f"✅ Reloaded: FP16={fp16_window}, INT8={int8_window}, Max={max_new_tokens}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
        