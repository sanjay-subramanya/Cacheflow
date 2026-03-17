from engine_manager import ModelManager
from ui.app_ui import build_ui
from config.settings import Config

config = Config()
manager = ModelManager(config)

demo = build_ui(
    chat_engine = manager.chat_engine, 
    update_fn = manager.update_and_reload, 
    current_config = config
)

if __name__ == "__main__":
    demo.launch(server_name = "0.0.0.0")
