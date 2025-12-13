from transformers import AutoModelForCausalLM, AutoTokenizer


class Validator:
    def __init__(self, name: str):
        self.name = name

    def validate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        raise NotImplementedError
