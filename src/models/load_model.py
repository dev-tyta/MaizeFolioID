from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model


def download_model(repo, file):
    repo = repo  # Hugging Face Repository - "Testys/MaizeFolioID"
    file = file  # File within Hugging Face repository "model_2.h5"

    model_2 = load_model(hf_hub_download(repo_id=repo, filename=file))
    return model_2
