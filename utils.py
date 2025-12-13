def to_safe_model_name(model_name: str) -> str:
    """
    Convert a model name to a safe model name for saving to the Hugging Face Hub.
    """
    return model_name.replace("/", "-")
