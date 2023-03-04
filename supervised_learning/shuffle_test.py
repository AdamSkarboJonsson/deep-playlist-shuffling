import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class SelfAttention(nn.Module):
    def __init__(self, in_features: int, embed: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features=in_features, out_features=embed)
        self.key = nn.Linear(in_features=in_features, out_features=embed)
        self.value = nn.Linear(in_features=in_features, out_features=embed)

    def forward(self, x: Tensor):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # scoring the query vectors against all key vectors
        scores = Q @ torch.swapdims(K,dim0=2, dim1=1)

        # computing the weights by a softmax operation
        scaled_scores = scores / K.shape[1] ** 0.5
        weights = nn.functional.softmax(scaled_scores, dim=-1)

        # computing the attention by a weighted sum of the value vectors
        attention = torch.bmm(weights, V)
        return attention, weights

class Shuffler(nn.Module):
    def __init__(self, N, D, tau=1):
        super(Shuffler, self).__init__()
        self.sa1 = SelfAttention(in_features=1, embed=32)
        self.sa2 = SelfAttention(in_features=32, embed=N)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-2)
        self.tau = tau

    def forward(self, x):
        B, N, D = x.shape
        x, w = self.sa1(x) # B, N, 4
        x, w = self.sa2(x) # B, N, N
        #x = torch.sigmoid(x)
        x = self.softmax1(x*self.tau)
        #x = self.softmax2(x*self.tau)
        return torch.reshape(x, (B, N, N))


class PermutationMatrixLoss(nn.Module):
    def __init__(self):
        super(PermutationMatrixLoss, self).__init__()

    def forward(self, p: Tensor):
        B, N, N = p.shape
        p_abs = torch.abs(p)
        p_squared = torch.pow(p, exponent=2)
        p_abs_sum_i = torch.sum(p_abs, dim=1, keepdim=True)
        p_abs_sum_j = torch.sum(p_abs, dim=2, keepdim=True)
        p_squared_sum_i = torch.sum(p_squared,dim=1, keepdim=True)
        p_squared_sum_j = torch.sum(p_squared,dim=2, keepdim=True)
        p_squared_root_sum_i = torch.pow(p_squared_sum_i,exponent=0.5)
        p_squared_root_sum_j = torch.pow(p_squared_sum_j,exponent=0.5)
        inner_diff_i = p_abs_sum_j-p_squared_root_sum_j
        inner_diff_j = p_abs_sum_i-p_squared_root_sum_i
        row_wise_loss = torch.sum(inner_diff_i, dim=1, keepdim=True)
        column_wise_loss = torch.sum(inner_diff_j, dim=2, keepdim=True)
        unsqueezed_loss = row_wise_loss+column_wise_loss
        loss = torch.squeeze(unsqueezed_loss)
        return loss



epochs = 100000

def train():
    data = Tensor([[[1],[0],[5]]])
    model = Shuffler(N=data.shape[1], D=data.shape[2])
    criteria = PermutationMatrixLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        out = model(data)

        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        #print(out)
        #print(torch.bmm(out, data))
        #print(torch.bmm(out, data))
        loss = criteria(out)
        loss.backward()



        optimizer.step()

        if epoch % 100 == 0:
            print(epoch, loss.item())
        #for name, param in model.named_parameters():
        #    print(name, param.grad)
    print(epoch, loss.item())

train()


