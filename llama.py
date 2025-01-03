from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import transformers
import torch
import os


def set_device(device: str | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        torch.set_default_device(device)


class Llama:
    """
    Llama is a wrapper class for the Hugging Face model "meta-llama/Llama-3.2-3B-Instruct".
    It initializes the model by loading the model and tokenizer from the Hugging Face model hub. Then, it builds a pipeline for text generation.
    """
    def __init__(self, device: str | None = None):
        set_device(device)
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        login(os.environ["HF_TOKEN"])

        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pipeline = transformers.pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=torch.get_default_device()
        )

    def __call__(self, system_prompt: str, user_prompt: str, previous_response: str = None, correction_prompt: str = None):
        """
        Generate a response from the model given the system prompt and user prompt.
        If the model response is incorrect, provide a correction prompt to the model and its previous response.
        The model is set with low top_p and temperature values for consistent and accurate responses.

        :param system_prompt: str
            System prompt to provide context and instructions to the Model.
        :param user_prompt: str
            User prompt to provide inputs to the Model.
        :param previous_response: str (optional)
            Previous model response with the system prompt and user prompt provided.
        :param correction_prompt: str (optional)
            Another user prompt to provide corrections to the previous response.
        :return: Model response.
        """
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if previous_response is not None and correction_prompt is not None:
            prompt.append({"role": "system", "content": previous_response})
            prompt.append({"role": "user", "content": correction_prompt})

        outputs = self.pipeline(
            prompt,
            do_sample=True,
            top_p=0.5,  # low top_p means for consistent and accurate responses
            temperature=0.1,  # low temperature means for deterministic responses
            max_new_tokens=256,
            truncation=True
        )
        return outputs[0]["generated_text"][-1]["content"]