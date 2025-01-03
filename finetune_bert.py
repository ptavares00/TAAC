from bert import CustomBERT
from artificial_dataset import ArtificialDataset

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Step 1: Define paths for the JSON files
DIRECTORY = '/home/paulo-bessa/Downloads'
#'/path/to/json/files'


# Step 2: Load and convert JSON files using the ArtificialDataset class
def load_and_convert_json(data_paths=DIRECTORY):
    dataset = ArtificialDataset(data_path=data_paths)
    #for file_path in data_paths:
        #with open(file_path, 'r') as f:
        #    data = json.load(f)
        
        # Assuming the response includes 'document', 'summary', and 'summary_word_tokenization'
        #document = data['content']['document']
        #summary = data['content']['summary']
        #summary_tokenization = data['content']['summary_word_tokenization']

        # Create a dataset object and append it
        #dataset.append(ArtificialDataset(data_path=file_path, tokenizer=tokenizer))

    return dataset


# Step 4: Train the DistilBERT model
def train_model(encodings):
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
        train_dataset=encodings,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":

    # Step 5: Load and convert the JSON files
    dataset = ArtificialDataset(data_path=DIRECTORY)

    # Step 6: Prepare the dataset for training
    #tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    #encodings = prepare_dataset_for_training(dataset,tokenizer)

    from IPython import embed; embed()

    # Step 7: Train the model
    #train_model(encodings)
