"""
llm_wrapper.py

Provides a wrapper class around Hugging Face Transformers models
for text generation, with an optional PEFT/LoRA adapter hook for
future fine-tuning support.
"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class TransformerLLM:
    """
    A wrapper around a Hugging Face causal language model that
    handles loading, tokenization, and text generation, with
    optional PEFT adapter support.

    Attributes:
        model_name (str): Hugging Face model identifier.
        device (str): Device to run inference on ('cpu' or 'cuda').
        model: Loaded causal LM model instance.
        tokenizer: Loaded tokenizer instance.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cpu"):
        """
        Initialize the wrapper and load the base model and tokenizer.

        Args:
            model_name (str): Hugging Face model repo ID. Defaults to
                a small, ungated model suitable for CPU inference.
            device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the tokenizer and base model from Hugging Face and
        move the model to the configured device.
        """
        print(f"Downloading/loading {self.model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)

    def load_peft_adapter(self, adapter_path: str) -> None:
        """
        Attach a PEFT/LoRA adapter to the base model, if one exists.
        Optional — only needed if fine-tuning is later required.

        Args:
            adapter_path (str): Path or HF repo ID of the PEFT adapter.
        """
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """
        Generate a text completion for the given prompt.

        Args:
            prompt (str): Input prompt text.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature (higher = more random).

        Returns:
            str: The generated text, excluding the input prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated_text

    def generate_chat(self, system_prompt: str, user_prompt: str,
                      max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        """
        Generate a response using the model's proper chat template,
        which significantly improves instruction-following compared
        to raw text prompting.

        Args:
            system_prompt (str): Instructions defining the assistant's role.
            user_prompt (str): The user-facing message, including context.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated_text

if __name__ == "__main__":
    print("Loading model...", flush=True)
    llm = TransformerLLM()
    print("Model loaded. Generating...", flush=True)
    response = llm.generate("What is Retrieval-Augmented Generation?")
    print("Response:", flush=True)
    print(response)

