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

# Step 3: Tokenization and preparation for training
def prepare_dataset_for_training(dataset,tokenizer):
    
    documents = [d['content']['document'] for d in dataset]
    summarys = [d['content']['summary'] for d in dataset]
    summary_labels = [d['content']['response']['unfaithful'] for d in dataset]
    labels = [d['content']['response']['word unfaithful labels'] for d in dataset]
    word_labels=[]

    tokenized_documents= tokenizer(documents, 
                            max_length=512,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt")
    
    tokenized_summarys = tokenizer(summarys, 
                            max_length=128,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt")

    for idx,label in enumerate(labels):    
        if label:
            word_labels.append([-100 if word_id is None else int(label[word_id][1]) for word_id in tokenized_summarys.word_ids(batch_index=idx)])
        else:
            word_labels.append([0] * len(tokenized_summarys['input_ids'][idx]))
        

    word_labels = [torch.tensor(word_label, dtype=torch.long) for word_label in word_labels]
    summary_labels = [torch.tensor(summary_label, dtype=torch.long) for summary_label in summary_labels]

    # Return the data as lists of tensors
    encodings = {
            "document_input_ids": tokenized_documents['input_ids'],#[t['input_ids'] for t in tokenized_documents],
            "document_attention_mask": tokenized_documents['attention_mask'],#[t['attention_mask'] for t in tokenized_documents],
            "summary_input_ids": tokenized_summarys['input_ids'],#[t['input_ids'] for t in tokenized_summarys],
            "summary_attention_mask": tokenized_summarys['attention_mask'],#[t['attention_mask'] for t in tokenized_summarys],
            'summary_label': summary_labels,  # Summary-level classification label
            "word_labels": word_labels  # Token-level classification labels
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
    dataset = load_and_convert_json(data_paths=DIRECTORY)

    # Step 6: Prepare the dataset for training
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encodings = prepare_dataset_for_training(dataset,tokenizer)

    from IPython import embed; embed()

    # Step 7: Train the model
    #train_model(encodings)
