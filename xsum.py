from datasets import load_dataset

def load_xsum():
    return load_dataset("xsum", trust_remote_code=True)