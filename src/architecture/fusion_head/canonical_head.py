
import ipdb
import torch 
import torch.nn as nn
from utils.utils import get_device
from src.architecture.backbone.backbone import CaptchaBackbone

device = get_device()

class CanonFusionHeadModel(nn.Module):
    def __init__(self, img_in_size=448, text_in_size=768, canon_size=300, shared_canon=True):
        super(CanonFusionHeadModel, self).__init__()

        # TODO experiment with this
        self.shared_canon = shared_canon

        self.img_proj = nn.Sequential(
            nn.Linear(img_in_size, canon_size),
            # nn.ReLU(),
            # nn.Linear(canon_size, canon_size),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_in_size, canon_size),
            # nn.ReLU(),
            # nn.Linear(canon_size, canon_size),
        )

        if self.shared_canon:
            self.shared_mlp = nn.Sequential(
            nn.Linear(canon_size, canon_size),
            nn.ReLU(),
            nn.Linear(canon_size, canon_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(canon_size*2, canon_size),
            nn.ReLU(),
            nn.Linear(canon_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequential(x)

class CanonFusionHead(CaptchaBackbone):
    def __init__(self, shared_canon=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(448)
        # self.shared_canon = shared_canon
        self.canon_model = CanonFusionHeadModel(shared_canon=shared_canon).to(device)

    def forward(self, x):
        im_embed, text_embed = x
        im_embed = torch.mean(im_embed, dim=1)  # collapse channels
        im_embed = self.layernorm(im_embed)

        # project to similar subspace
        img_emb_hat = self.canon_model.img_proj(im_embed)
        text_emb_hat = self.canon_model.text_proj(text_embed)

        if self.canon_model.shared_canon:
            img_canon = self.canon_model.shared_mlp(img_emb_hat)
            text_canon = self.canon_model.shared_mlp(text_emb_hat)
            comb_hat = torch.cat((img_canon, text_canon), dim=1)

        else:
            comb_hat = torch.cat((img_emb_hat, text_emb_hat), dim=1)
        # print(f"using shared canonical model: {self.canon_model.shared_canon}")
        # ipdb.set_trace()
        
        return self.canon_model.classifier(comb_hat)
