from transformers import DistilBertModel, DistilBertTokenizerFast
import torch.nn.functional as F
from torch import nn
import torch


class CustomBERT(nn.Module):
    def __init__(self, max_size_summary_tokens=128):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.summary_classifier = nn.Linear(self.distilbert.config.hidden_size, 1)  # 2 classes (faithful / not faithful)
        self.token_classifier = nn.Linear(self.distilbert.config.hidden_size, 1)  # 2 classes (faithful / not faithful)
        self.max_size_summary_tokens = max_size_summary_tokens

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        cls_hidden_state = hidden_states[:, 0, :]  # [CLS] token at position 0
        summary_logits = self.summary_classifier(cls_hidden_state)

        # Identify [SEP] token positions for each sequence in the batch
        sep_token_id = self.tokenizer.sep_token_id
        sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=False)
        token_logits = [
            self._forward_token_classification(
                hidden_states[i].unsqueeze(0),  # (1, sequence_length, hidden_size)
                sep_indices[sep_indices[:, 0] == i][:, 1]
            ) for i in range(batch_size)
        ]
        token_logits = torch.stack(token_logits, dim=0)  # (batch_size, max_size, 1)

        return summary_logits, token_logits

    def _forward_token_classification(self, hidden_states, sep_positions):
        summary_start = sep_positions[0] + 1  # Start of the summary tokens
        summary_hidden_states = hidden_states[:, summary_start:-1, :]  # Only summary tokens, (summary_length, hidden_size)
        token_logits = self.token_classifier(summary_hidden_states)  # (1, summary_length, 1)
        token_logits = F.pad(token_logits, (0, 0, 0, self.max_size_summary_tokens - token_logits.size(1)), mode='constant', value=0)  # (1, max_size, 1)
        return token_logits.squeeze(0)  # (max_size, 1)


if __name__ == "__main__":
    model = CustomBERT(128)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    documents = [
        "This is my very first document. Please don't make fun of me!!!",
        "Another example document. It has different content."
    ]
    summaries = [
        "This is my first summary.",
        "This is another summary example."
    ]
    tokenized = tokenizer(documents, summaries, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    s_logits, t_logits = model(input_ids, attention_mask)
    pass