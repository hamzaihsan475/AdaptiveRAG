"""
text_generator.py

Handles text generation using an already-loaded model and
tokenizer. Single responsibility: generation only, not loading.
"""


class TextGenerator:
    """
    Generates text completions using a pre-loaded model and
    tokenizer, either via raw prompting or the model's chat template.

    Attributes:
        model: A loaded causal LM model instance.
        tokenizer: A loaded tokenizer instance.
        device (str): Device the model is on.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        """
        Initialize the generator with an already-loaded model.

        Args:
            model: A loaded Hugging Face causal LM.
            tokenizer: The matching tokenizer.
            device (str): 'cpu' or 'cuda'.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.2) -> str:
        """
        Generate a text completion for a raw prompt.

        Args:
            prompt (str): Input prompt text.
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def generate_chat(self, system_prompt: str, user_prompt: str,
                       max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        """
        Generate a response using the model's chat template.

        Args:
            system_prompt (str): Instructions defining the assistant's role.
            user_prompt (str): The user-facing message, including context.
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )