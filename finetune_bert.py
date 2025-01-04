from bert import CustomBERT
from artificial_dataset import ArtificialDataset
import random
import os
import glob
from datasets import Dataset, DatasetDict
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Step 1: Define paths for the JSON files
DIRECTORY = '/home/paulo-bessa/Downloads/faithfulness_dataset_filtered'

def split_data(dataset_paths=DIRECTORY, train_size=0.8, val_size=0.1):
    """
    Splits the dataset into train, validation, and test datasets based on paths.
    """
    # Shuffle the data
    json_files=glob.glob(os.path.join(dataset_paths, "*.json"))
    random.seed(42)
    #random.shuffle(json_files) #Commented out to avoid shuffling the data

    # Calculate sizes
    train_end = int(train_size * len(json_files))
    val_end = int((train_size + val_size) * len(json_files))

    # Split the data
    train_data = json_files[:train_end]
    val_data = json_files[train_end:val_end]
    test_data = json_files[val_end:]

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    return train_data, val_data, test_data


def convert_to_dataset_dict(dataset_paths):
    """
    Converts the JSON files into a DatasetDict object.
    """

    train_data, val_data, test_data=split_data(dataset_paths, train_size=0.8, val_size=0.1)

    return DatasetDict({
                        'train': ArtificialDataset(data_paths=train_data),
                        'validation': ArtificialDataset(data_paths=val_data),
                        'test': ArtificialDataset(data_paths=test_data)
                        })

# Step 2: Define the compute_metrics function
############### Not sure if This function is working properly ################
def compute_metrics(predictions, labels):
    """
    Computes the precision, recall, F1, and accuracy of the model.
    Assumes binary classification (adjust for multi-class if needed).
    """
    # For binary classification, convert predictions to binary (0 or 1)
    predictions = (predictions >= 0.5).astype(int)  # If using probabilities, threshold at 0.5
    
    # Convert labels to binary (0 or 1) if needed
    labels = labels.astype(int)

    # Compute precision, recall, F1 score, and accuracy using scikit-learn
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

# Step 2: Train the DistilBERT model
def train_model(tokenized_datasets):
    """
    Trains the DistilBERT model on the tokenized datasets.
    """

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Set up Trainer arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics #Not quite sure if its working properly
        
        #Dont need to pass tokenizer and data_collator as we are using tokenizer in artifical_dataset.py
        #data_collator=data_collator,
        #tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":

    # Step 5: Load and convert the JSON files

    dataset_dict=convert_to_dataset_dict(dataset_paths=DIRECTORY)
    
    # Step 7: Train the model
    ############### Not sure if the model is training properly, but at least it runs ################
    #Only could run the model with the labels at summary level
    #Not quite sure how to train the model with the labels at token level (didnt had time to test it, enough)
        # It was always giving me an error when trying to pass the labels at token level
        
    train_model(dataset_dict)

    from IPython import embed; embed()

