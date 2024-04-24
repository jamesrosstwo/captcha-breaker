import os
import ipdb
import torch
import torch.nn as nn
from utils.utils import get_device
from src.architecture.backbone.backbone import CaptchaBackbone
from transformers import AlbertModel
from transformers import AlbertTokenizer


device = get_device()

class AlbertEmbeddingBackbone(CaptchaBackbone):
    def __init__(self):
        super().__init__()
        self.albert = AlbertModel.from_pretrained('albert-base-v2').to(device)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    def forward(self, text):
        text_tokenized = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512, add_special_tokens=True)
        
        # TODO get the input ids and attention mask from here.
        input_ids = text_tokenized['input_ids'].to(device)
        attention_mask = text_tokenized['attention_mask'].to(device)

        # Squeeze out the extra dimension because it needs to be shape (batch_size, seq_length)
        input_ids = torch.squeeze(input_ids, dim=1).to(device)
        attention_mask = torch.squeeze(attention_mask, dim=1).to(device)

        # Getting outputs from ALBERT
        outputs = self.albert(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        # Use the hidden state of first CLS token as the embedding (one CLS token per sequence)
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding
