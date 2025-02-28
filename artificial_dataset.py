import json
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast
from torch.nn import functional as F

class ArtificialDataset(Dataset):
    def __init__(self, data_paths):
        self.json_files = data_paths
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def __getitem__(self, idx):
        with open(self.json_files[idx], 'r') as file:
            json_file = json.load(file)
        
        return self.prepare_dataset_for_training(json_file)
    

    def __len__(self) -> int:
        """
        Returns the number of JSON files in the dataset.
        """
        return len(self.json_files)
    
    # This function will take a JSON file and prepare the data for training
    def prepare_dataset_for_training(self,json_file):
        """
        Prepares the dataset for training by tokenizing the document and summary.
        """

        document = json_file['content']['document']
        summary = json_file['content']['summary']
        summary_word_tokenization = json_file['content']['summary_word_tokenization']
        summary_label = json_file['content']['response']['unfaithful'] 
        label = json_file['content']['response']['word unfaithful labels'] 

        # Tokenize the document and summary
        tokenized_documents= self.tokenizer(document, summary,
                                max_length=512, #maximum allowd by distilbert
                                padding='max_length',
                                truncation=True,
                                return_tensors="pt")        

        # Tokenize the summary at the word level 
        tokenized_summarys = self.tokenizer(summary_word_tokenization, 
                                max_length=128,
                                padding='max_length',
                                truncation=True,
                                return_tensors="pt",
                                is_split_into_words=True)
        
        # Get the word IDs from the tokenized summary
        # This will be used to create the token-level labels
        # It garantees that the labels are aligned with the tokenized subwords 
        word_ids = tokenized_summarys.word_ids()

        if label:
        # Convert the labels to binary values and -100 for the special tokens
            word_labels=[-100 if word_id is None else int(str(label[word_id][1]).lower() == 'true') for word_id in word_ids]
        else:
            word_labels=[0] * 128 ## padding with zeros if no labels are available
            
        word_labels = torch.tensor(word_labels, dtype=torch.long)
        summary_label = torch.tensor(summary_label, dtype=torch.long)

        encodings = {
            "input_ids": tokenized_documents['input_ids'].squeeze(0),  # Document and Summary input IDs
            "attention_mask": tokenized_documents['attention_mask'].squeeze(0),  # Document and Summary attention masks
            "labels": summary_label.squeeze(0).unsqueeze(-1),  # Summary-level classification label
            "word_labels": word_labels.squeeze(0).unsqueeze(-1)  # Token-level classification labels
        }
        
        return encodings

if __name__ == "__main__":
    import os
    import glob
    
    data_paths=glob.glob(os.path.join("/home/paulo-bessa/Downloads/dataset_max_token_size_512", "*.json"))
    dataset = ArtificialDataset(data_paths=data_paths)
 
    from IPython import embed; embed()

    