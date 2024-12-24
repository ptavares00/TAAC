from torch import nn
from transformers import DistilBertModel


class CustomBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.summary_classifier = nn.Linear(self.distilbert.config.hidden_size, 2)  # 2 classes (faithful / not faithful)
        self.word_classifier = nn.Linear(self.distilbert.config.hidden_size, 2)  # 2 classes (faithful / not faithful)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)

        cls_representation = hidden_states[:, 0, :]  # [CLS] token is at position 0
        summary_logits = self.summary_classifier(cls_representation)

        word_logits = self.word_classifier(hidden_states)

        return summary_logits, word_logits