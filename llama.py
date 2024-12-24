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

    def __call__(self, system_prompt: str, user_prompt: str):
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        outputs = self.pipeline(
            prompt,
            do_sample=True,
            top_p=0.5,  # low top_p means for consistent and accurate responses
            temperature=0.1,  # low temperature means for deterministic responses
            max_new_tokens=128,
            truncation=True
        )
        return outputs[0]["generated_text"][-1]["content"]