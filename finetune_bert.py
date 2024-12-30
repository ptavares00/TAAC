from bert import CustomBERT
from artificial_dataset import ArtificialDataset

import os
import glob
import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Step 1: Define paths for the JSON files
directory = '/path/to/json/files'

# Use glob to get all .json files in the directory
data_paths = glob.glob(os.path.join(directory, "*.json"))

# Step 2: Load and convert JSON files using the ArtificialDataset class
def load_and_convert_json(data_paths):
    dataset = []
    for file_path in data_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Assuming the response includes 'document', 'summary', and 'summary_word_tokenization'
        document = data['content']['document']
        summary = data['content']['summary']
        summary_tokenization = data['content']['summary_word_tokenization']

        # Create a dataset object and append it
        dataset.append(ArtificialDataset(document=document, summary=summary, summary_tokenization=summary_tokenization))

    return dataset

# Step 3: Tokenization and preparation for training
def prepare_dataset_for_training(dataset):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Prepare input texts (documents and summaries)
    inputs = [d.document for d in dataset]
    summaries = [d.summary for d in dataset]

    # Tokenize the inputs and summaries
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    summary_encodings = tokenizer(summaries, truncation=True, padding=True, max_length=128)

    # Combine into a final dataset format
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': summary_encodings['input_ids']
    }

    return encodings

# Step 4: Train the DistilBERT model
def train_model(encodings):
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Set up Trainer arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encodings,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    # Step 5: Load and convert the JSON files
    dataset = load_and_convert_json(data_paths)

    # Step 6: Prepare the dataset for training
    encodings = prepare_dataset_for_training(dataset)

    # Step 7: Train the model
    train_model(encodings)
