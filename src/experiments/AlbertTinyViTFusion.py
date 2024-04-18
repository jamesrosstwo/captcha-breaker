import torch
import os
import sys
import torch.nn as nn
# Add the root directory to the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.utils import get_device
from src.models.tiny_vit.tiny_vit import TinyViT, tiny_vit_11m_224
from src.utils.train_utils import train
from transformers import AlbertModel
from transformers import AlbertTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for id_dir in sorted(os.listdir(self.data_dir)):
            id_path = os.path.join(self.data_dir, id_dir)
            if os.path.isdir(id_path):
                question_path = os.path.join(id_path, 'question.txt')
                with open(question_path, 'r') as file:
                    question_text = file.read().strip().split('\n')[0] # TODO, we do this as a quick fix for the way question.txt 
                    #       is set up, since it contains more than just the question.
                   
                    question_tokenized = self.tokenizer(question_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512, add_special_tokens=True)

                for label_dir in ['positive', 'negative']:
                    label_path = os.path.join(id_path, label_dir)
                    label = 1 if label_dir == 'positive' else 0
                    for image_name in os.listdir(label_path):
                        image_path = os.path.join(label_path, image_name)
                        samples.append((image_path, question_tokenized, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, question_tokenized, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, question_tokenized, torch.tensor(label).unsqueeze(0).float().to(device)

# Transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])


# The class where we define what we do with the text and image embeddings extracted
class FusionHead(nn.Module):
    def __init__(self, input_size):
        super(FusionHead, self).__init__()

        # TODO experiment with this
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)


class AlbertEmbeddingModel(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert.to(device)

    def forward(self, input_ids, attention_mask=None):
        # Squeeze out the extra dimension because it needs to be shape (batch_size, seq_length)
        input_ids = torch.squeeze(input_ids, dim=1).to(device)
        attention_mask = torch.squeeze(attention_mask, dim=1).to(device)

        # Getting outputs from ALBERT
        outputs = self.albert(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        # Use the hidden state of first CLS token as the embedding (one CLS token per sequence)
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding


class Albert_TinyViT_CaptchaBreaker(nn.Module):
    def __init__(self, tiny_vit, albert: AlbertEmbeddingModel):
        super(Albert_TinyViT_CaptchaBreaker, self).__init__()
        self.tiny_vit = tiny_vit.to(device)  # Produces embeddings of shape (1, 448)
        self.albert = albert.to(device)   # Produces embeddings of 768
        # Initialize the fusion head with the output dimensions of tiny_vit and albert
        self.fusion_head = FusionHead(448+768).to(device)
        
        # Freeze the parameters in the Albert model
        for param in self.albert.parameters():
            param.requires_grad = False
        
        # Freeze the parameters in the ViT model
        for param in self.tiny_vit.parameters():
            param.requires_grad = False


    def forward(self, x_img, x_text):
        input_ids = x_text['input_ids'].to(device)
        attention_mask = x_text['attention_mask'].to(device)

        # Get text and image embeddings using their respective models
        img_embed = self.tiny_vit(x_img)
        text_embed = self.albert(input_ids, attention_mask)
        # Concatenate the embeddings
        combined_embed = torch.cat((img_embed, text_embed), dim=1)

        # Forward pass through the fusion head
        out = self.fusion_head(combined_embed)
        return out

# Getting our device dynamically to support training on every platform
device = get_device()

# Setting up the vision and text feature extractors 
tiny_vit = tiny_vit_11m_224(pretrained=True).to(device)
albert = AlbertModel.from_pretrained('albert-base-v2').to(device)
albert_emb_model = AlbertEmbeddingModel(albert).to(device)  # Albert model returning only the embeddings

# The combined Captcha Breaker model
model = Albert_TinyViT_CaptchaBreaker(tiny_vit, albert_emb_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Create the dataset instances for training and validation
train_dataset = CaptchaDataset(data_dir='src/dataset/data/train', tokenizer=albert_tokenizer, transform=transform)
val_dataset = CaptchaDataset(data_dir='src/dataset/data/val', tokenizer=albert_tokenizer, transform=transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

train(model, optimizer, train_loader, val_loader, torch.nn.BCELoss(), epochs=10, device=device)
