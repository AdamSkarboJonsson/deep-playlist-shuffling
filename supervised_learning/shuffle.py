from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from deep_shuffling.dataset import create_playlist_dataset, PlaylistDataset
from deep_shuffling.neuralsort import NeuralSort
from deep_shuffling.softsort import SoftSort
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('qtagg')
n_batch_size = 1
epochs = 32
maximum_playlist_length = 1024
n_embed = 16
n_heads = 8
device = torch.device('cuda')
torch.cuda.manual_seed(1337)
torch.random.manual_seed(1337)


def project_p(P_hat):
    dim = 512
    P = torch.zeros_like(P_hat, device='cuda')
    b_idx = torch.arange(1).repeat([1, dim]).view(dim, 1).transpose(
        dim0=1, dim1=0).flatten().type(torch.cuda.LongTensor)
    r_idx = torch.arange(dim).repeat(
        [1, 1]).flatten().type(torch.cuda.LongTensor)
    c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
    brc_idx = torch.stack((b_idx, r_idx, c_idx))

    P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
    P_hat = (P - P_hat).detach() + P_hat
    return P_hat


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, in_features: int, n_embed: int, n_heads: int):
        super().__init__()
        self.query = nn.Linear(in_features=in_features, out_features=n_embed, device=device)
        self.key = nn.Linear(in_features=in_features, out_features=n_embed, device=device)
        self.value = nn.Linear(in_features=in_features, out_features=n_embed, device=device)
        self.multiheadattention = nn.MultiheadAttention(embed_dim=n_embed,
                                                        num_heads=n_heads,
                                                        dropout=0,
                                                        batch_first=True,
                                                        device=device)

    def forward(self, x, mask):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = self.multiheadattention(query=q,
                                    key=k,
                                    value=v,
                                    key_padding_mask=mask,
                                    need_weights=False,
                                    attn_mask=None,
                                    average_attn_weights=True)
        return x


class ShuffleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = MultiheadAttentionBlock(in_features=2, n_embed=n_embed, n_heads=n_embed // 2)
        self.relu1 = nn.ReLU()
        self.b2 = MultiheadAttentionBlock(in_features=n_embed, n_embed=n_embed, n_heads=n_embed // 2)
        self.relu2 = nn.ReLU()
        self.b3 = MultiheadAttentionBlock(in_features=n_embed, n_embed=1, n_heads=1)
        self.sort = SoftSort()#NeuralSort(tau=1)
        self.l1 = nn.Linear(in_features=2, out_features=n_embed, bias=False, device=device)
        self.l2 = nn.Linear(in_features=n_embed, out_features=1, bias=False, device=device)


    def forward(self, inp: dict[str, torch.tensor]):
        # x: {"constant", "must_vary"}
        xc = inp["constant"]
        mask: torch.Tensor = inp["mask"]
        #x, _ = self.b1(xc, mask)
        x = self.l1(xc)
        x = self.relu1(x)
        x = self.l2(x)
        #x, _ = self.b2(x, mask)
        #x = self.relu2(x)
        #x, _ = self.b3(x, mask)
        B, N, _ = x.shape
        x = torch.reshape(x, shape=(B, N))
        x = torch.masked_fill(x, mask=mask, value=-torch.inf)
        x = self.sort(x)
        return x


class PermutationMatrixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, M: torch.Tensor):
        B, N, N = M.shape
        M2 = torch.square(M)
        M_abs = torch.abs(M)
        column_loss = torch.sum(torch.sum(M_abs, dim=2, keepdim=True) - torch.pow(torch.sum(M2, dim=2, keepdim=True), exponent=0.5), dim=1, keepdim=True)
        row_loss = torch.sum(torch.sum(M_abs, dim=1, keepdim=True) - torch.pow(torch.sum(M2, dim=1, keepdim=True), exponent=0.5), dim=2, keepdim=True)
        loss = torch.squeeze(column_loss + row_loss)/N
        return loss


class ShuffleLoss(nn.Module):
    def __init__(self, lambd: float):
        super(ShuffleLoss, self).__init__()
        self.avg_pooling = torch.nn.AvgPool2d(kernel_size=(3, 1))
        self.permutation_matrix_loss = PermutationMatrixLoss()
        self.lambd = lambd

    def forward(self, permutation_matrix, features):
        features_sorted = torch.bmm(permutation_matrix, features[:, :])
        avg_feats = torch.sum(features_sorted, dim=-2)
        shifted_features = torch.roll(features_sorted, -1, -2)
        pooling = self.avg_pooling(features_sorted)
        noise_squared_diff = (features_sorted[:, :-1, :] - shifted_features[:, :-1, :]) ** 2
        noise_loss = torch.sum(noise_squared_diff)**0.5
        pooling_squared_diff = (pooling - avg_feats) ** 2
        global_level_loss = torch.sum(pooling_squared_diff)
        #permutation_matrix_loss = self.permutation_matrix_loss(permutation_matrix)
        loss = noise_loss + global_level_loss# + self.lambd*permutation_matrix_loss
        return loss


def train(model: nn.Module, dataset: PlaylistDataset):
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1)
    criterion = ShuffleLoss(lambd=1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.01, )
    torch.autograd.set_detect_anomaly(True)
    for i in range(1000):
        for playlist in data_loader:
            criterion.zero_grad()
            out: torch.Tensor = model(playlist)
            # print(torch.argmax(out[0, 0, :]))
            loss = criterion(out, playlist["constant"])
            print(loss.item())
            loss.backward()
            optimizer.step()
    return model


def apply_model(playlist, model):
    n = playlist["n"]
    B, N, D = playlist["constant"].shape
    p = model(playlist)
    p_star = project_p(p)[0, :, :]
    print(p_star)
    d_star = p_star @ playlist["constant"][0, :n, :]

    d_line = (d_star[:, di] for di in range(D))
    for line in d_line:
        l = line.tolist()
        plt.scatter(list(range(n)), l)
    plt.show()


if __name__ == "__main__":
    model = ShuffleModel()
    dataset = create_playlist_dataset()
    model = train(model=model, dataset=dataset)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1)
    for playlist in data_loader:
        apply_model(playlist, model=model)
