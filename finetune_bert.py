import random
import os
import glob
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from bert import CustomBERT
from artificial_dataset import ArtificialDataset

# Step 1: Define paths for the JSON files
DIRECTORY = 'c:/Users/nunom/Downloads/testing'

# Step 2: Split the dataset into train, validation, and test datasets
def split_data(dataset_paths=DIRECTORY, train_size=0.8, val_size=0.1):
    json_files = glob.glob(os.path.join(dataset_paths, "*.json"))
    random.seed(42)
    random.shuffle(json_files)

    train_end = int(train_size * len(json_files))
    val_end = int((train_size + val_size) * len(json_files))

    train_data = json_files[:train_end]
    val_data = json_files[train_end:val_end]
    test_data = json_files[val_end:]

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    return train_data, val_data, test_data

# Step 3: Custom training loop
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=3):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            summary_labels = batch['labels'].to(device)
            word_labels = batch.get('word_labels', torch.empty(0)).to(device)

            optimizer.zero_grad()

            # Forward pass
            summary_logits, word_logits = model(input_ids, attention_mask)

            # Compute losses
            summary_loss = criterion(summary_logits, summary_labels)
            token_loss = (
                criterion(word_logits.view(-1, 2), word_labels.view(-1))
                if word_labels.numel() > 0 else 0
            )
            loss = summary_loss + token_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

        # Validate the model
        validate_model(model, val_loader, device)

def validate_model(model, val_loader, device):
    model.eval()
    all_summary_preds = []
    all_summary_labels = []
    all_word_preds = []
    all_word_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            summary_labels = batch['labels'].to(device)
            word_labels = batch.get('word_labels', torch.empty(0)).to(device)

            # Forward pass
            summary_logits, word_logits = model(input_ids, attention_mask)

            # Summary-level predictions
            summary_preds = torch.argmax(summary_logits, dim=-1)
            
            # Token-level predictions
            word_preds = torch.argmax(word_logits, dim=-1)

            # Collect predictions and labels for the summary task
            all_summary_preds.extend(summary_preds.cpu().numpy())
            all_summary_labels.extend(summary_labels.cpu().numpy())

            # Collect predictions and labels for the token task (word-level)
            if word_labels.numel() > 0:  # Check if word_labels are available
                # Flatten token-level predictions and labels to make metrics calculation easier
                valid_mask = word_labels != -100  # Mask to ignore padding tokens
                all_word_preds.extend(word_preds[valid_mask].cpu().numpy())
                all_word_labels.extend(word_labels[valid_mask].cpu().numpy())

    # Calculate summary-level metrics
    summary_accuracy = accuracy_score(all_summary_labels, all_summary_preds)
    summary_precision = precision_score(all_summary_labels, all_summary_preds, average='binary')
    summary_recall = recall_score(all_summary_labels, all_summary_preds, average='binary')
    summary_f1 = f1_score(all_summary_labels, all_summary_preds, average='binary')

    # Calculate token-level metrics (word-level)
    if all_word_labels:
        word_accuracy = accuracy_score(all_word_labels, all_word_preds)
        word_precision = precision_score(all_word_labels, all_word_preds, average='binary')
        word_recall = recall_score(all_word_labels, all_word_preds, average='binary')
        word_f1 = f1_score(all_word_labels, all_word_preds, average='binary')
        print(f"Token-level - Accuracy: {word_accuracy:.4f}, Precision: {word_precision:.4f}, Recall: {word_recall:.4f}, F1: {word_f1:.4f}")
    else:
        word_accuracy = word_precision = word_recall = word_f1 = None
        print("No word labels available for evaluation.")

    # Print summary-level metrics
    print(f"Summary-level - Accuracy: {summary_accuracy:.4f}, Precision: {summary_precision:.4f}, Recall: {summary_recall:.4f}, F1: {summary_f1:.4f}")

    return {
        "summary_accuracy": summary_accuracy,
        "summary_precision": summary_precision,
        "summary_recall": summary_recall,
        "summary_f1": summary_f1,
        "word_accuracy": word_accuracy,
        "word_precision": word_precision,
        "word_recall": word_recall,
        "word_f1": word_f1,
    }


def show_example_predictions(model, test_loader, device, example_index=0):
    model.eval()

    with torch.no_grad():
        # Get the batch from the test loader
        batch = test_loader.dataset[example_index]
        
        input_ids = batch['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
        attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
        summary_labels = batch['labels'].unsqueeze(0).to(device)
        word_labels = batch.get('word_labels', torch.empty(0)).unsqueeze(0).to(device)

        # Forward pass
        summary_logits, word_logits = model(input_ids, attention_mask)

        # Get predicted values
        summary_pred = torch.argmax(summary_logits, dim=-1).cpu().numpy()[0]
        word_pred = torch.argmax(word_logits, dim=-1).cpu().numpy()[0]

        # Get the true labels for the example
        summary_true = summary_labels.cpu().numpy()[0]
        word_true = word_labels.cpu().numpy()[0] if word_labels.numel() > 0 else []

        # Print the summary prediction and true label
        print(f"Example {example_index} Summary - Predicted: {summary_pred}, True: {summary_true}")

        # Print token predictions and true labels
        print(f"Example {example_index} Tokens - Predicted: {word_pred}, True: {word_true}")

        return summary_pred, summary_true, word_pred, word_true

# Step 5: Main script
if __name__ == "__main__":
    train_files, val_files, test_files = split_data()

    train_dataset = ArtificialDataset(train_files)
    val_dataset = ArtificialDataset(val_files)
    test_dataset = ArtificialDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = CustomBERT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)  # Handle padding tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, val_loader, optimizer, criterion, device)

    show_example_predictions(model, test_loader, device, example_index=0)



