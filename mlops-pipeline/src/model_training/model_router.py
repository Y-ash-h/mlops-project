from src.utils.data_router import detect_data_type
from src.model_training.train_text import train_text_model
from src.model_training.train_image import train_image_model
from src.model_training.train_audio import train_audio_model
from src.model_training.train_tabular import train_tabular_model

def route_training(data_dir: str):
    dtype = detect_data_type(data_dir)

    if dtype == "text":
        return train_text_model(data_dir)

    if dtype == "image":
        return train_image_model(data_dir)

    if dtype == "audio":
        return train_audio_model(data_dir)

    return train_tabular_model(data_dir)
