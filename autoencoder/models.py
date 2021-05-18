import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json
import os
import numpy as np

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, seqlen, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.seqlen = seqlen
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)

        # Encoder and Decoder
        self.encoder = nn.Linear(seqlen*emsize, nhidden)
        self.decoder = nn.Linear(nhidden, seqlen*nhidden)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)
        self.fc_mean = nn.Linear(nhidden, nhidden)
        self.fc_var = nn.Linear(nhidden, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)
        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def forward(self, indices):
        hidden = self.encode(indices)
        decoded = self.decode(hidden)
        return decoded

    def encode(self, indices, noise=None):
        embeddings = self.embedding(indices)
        embeddings = embeddings.view([-1, self.seqlen*self.emsize])
        # Encode
        hidden = self.encoder(embeddings)
        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)        
        # For older versions of PyTorch use:
        #hidden = torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))     
        return hidden

    def decode(self, hidden):
        # batch x hidden
        output = self.decoder(hidden)
        output = output.view(-1, self.seqlen, self.nhidden)
        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output)
        decoded = decoded.view(-1, self.ntokens)
        #vals, decoded = torch.max(decoded, 1)
        return decoded
