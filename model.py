import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from transformers import SamModel


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, latent_size))
        # decoder
        self.decoder = nn.Sequential(nn.Linear(latent_size, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, input_size))

    def forward(self, x):
        x = x.float()
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.gnn_embed = nn.Embedding.from_pretrained(torch.load("./GNN/output/gnn_embed.pt"), freeze=True)

        self.image_model = SamModel.from_pretrained("facebook/sam-vit-base").vision_encoder

        self.auto_encoder = AutoEncoder(config['input_size'], config['hidden_size'], config['latent_size'])
        self.auto_encoder.load_state_dict(
            torch.load("./AutoEncoder/output/Autoencoder_noiseX_model_state_dict_ep6.pth")
        )
        self.auto_encoder = self.auto_encoder.encoder

        for param in self.image_model.parameters():
            param.requires_grad = False

        for param in self.auto_encoder.parameters():
            param.requires_grad = False

        self.linear1 = nn.Linear(684, 512)
        self.linear2 = nn.Linear(512, 267)

    def forward(self, feature, image, ppt_idx):
        # auto encoder : 300
        h1 = self.auto_encoder(feature)

        # image encoder : 256
        h2 = self.image_model(**image).last_hidden_state
        h2 = F.adaptive_avg_pool2d(h2, (1, 1)).squeeze()

        # graph neural network : 128
        h3 = self.gnn_embed(ppt_idx)

        # feature fusion
        h = torch.concat([h1, h2, h3], dim=1)

        # prediction layer
        h = self.linear1(h)
        h = self.linear2(h)

        return h
