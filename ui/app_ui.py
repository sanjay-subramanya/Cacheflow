import gradio as gr

def build_ui(chat_engine, update_fn, current_config):
    """Gradio UI for chat interface"""
    def chat_fn(message, history):
        if len(history) == 0 and message:
            chat_engine.clear_history()

        full_response = ""
        last_telemetry = None
        
        for partial_response, telemetry in chat_engine.respond(message):
            full_response = partial_response
            last_telemetry = telemetry
            
            stats = f"""
                ---
                **KV Cache Telemetry**
                FP16 tokens: {last_telemetry.fp_tokens}
                INT8 tokens: {last_telemetry.int8_tokens}
                Total tokens: {last_telemetry.lifetime_tokens}
                Memory saved: {last_telemetry.saved_mb:.2f} MB
            """
            yield full_response + stats

    demo = gr.ChatInterface(
    fn=chat_fn,
    title="Cacheflow",
    description="""
        Chat with a stateful LLM with Hierarchical KV Cache Compression. Cacheflow maintains conversation context across turns while keeping memory usage bounded through intelligent cache management. 
        The system stores recent tokens in high precision for quality, compresses older tokens to reduce footprint, and evicts the oldest when configurable windows are exceeded.
        
        How it works:  
        • Recent tokens stored in FP16 (2 bytes per element)  
        • Older tokens quantized to INT8 (1 byte per element)  
        • Cache organized as sliding windows with FIFO eviction  
        • Memory usage remains capped regardless of conversation length  
        • Compressed tensors are fed directly into attention computations for subsequent tokens  
        
        Live telemetry displays current cache composition (FP16/INT8 token usage) and estimated memory savings compared to uncompressed inference. 
        Use the configuration panel below to experiment with different cache window sizes to see how they affect memory usage and response quality.
        """,
    )
    
    with demo:
        gr.Markdown("---")
        gr.Markdown("## ⚙️ Configuration Panel")
        
        with gr.Row():
            fp16_slider = gr.Slider(
                minimum=64, maximum=512, step=32,
                value=current_config.fp16_window,
                label="FP16 Window (high quality)"
            )
            
            int8_slider = gr.Slider(
                minimum=128, maximum=1024, step=64,
                value=current_config.int8_window,
                label="INT8 Window (compressed)"
            )
            
            max_tokens_slider = gr.Slider(
                minimum=50, maximum=300, step=25,
                value=current_config.max_new_tokens,
                label="Max new tokens"
            )
        
        update_btn = gr.Button("🔄 Update Configuration", variant="primary", size="lg")
        config_status = gr.Textbox(label="Status", interactive=False)
        
        update_btn.click(
            fn=update_fn,
            inputs=[fp16_slider, int8_slider, max_tokens_slider],
            outputs=[config_status]
        )    
    
    return demo