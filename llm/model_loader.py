"""
model_loader.py

Handles loading a Hugging Face causal language model and
tokenizer, with optional PEFT/LoRA adapter loading. Single
responsibility: loading only, not generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ModelLoader:
    """
    Loads a Hugging Face causal LM and tokenizer, with an optional
    PEFT adapter attached afterward.

    Attributes:
        model_name (str): Hugging Face model identifier.
        device (str): Device to load the model onto.
        model: The loaded model instance.
        tokenizer: The loaded tokenizer instance.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cpu"):
        """
        Initialize and load the base model and tokenizer.

        Args:
            model_name (str): Hugging Face model repo ID.
            device (str): 'cpu' or 'cuda'.
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)

    def load_peft_adapter(self, adapter_path: str) -> "ModelLoader":
        """
        Attach a PEFT/LoRA adapter to the loaded model.

        Args:
            adapter_path (str): Path or HF repo ID of the PEFT adapter.

        Returns:
            ModelLoader: self, to allow method chaining.
        """
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        return self