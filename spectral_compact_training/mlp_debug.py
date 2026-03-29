import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_qr(M):
    dev = M.device
    Q, R = torch.linalg.qr(M)
    return (Q * torch.sign(torch.diag(R))).to(dev)

class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        U = torch.randn(in_features, rank) / math.sqrt(in_features)
        V = torch.randn(out_features, rank) / math.sqrt(out_features)
        self.U = nn.Parameter(safe_qr(U)[:, :rank])
        self.V = nn.Parameter(safe_qr(V)[:, :rank])
        self.s = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return (x @ self.U) * self.s @ self.V.T + self.bias

    @torch.no_grad()
    def retract(self):
        self.U.data = safe_qr(self.U.data)[:, :self.rank]
        self.V.data = safe_qr(self.V.data)[:, :self.rank]

class DenseMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SpectralMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, rank=16):
        super().__init__()
        self.fc1 = SpectralLinear(in_dim, hidden, rank=min(rank, in_dim))
        self.fc2 = SpectralLinear(hidden, hidden, rank=rank)
        self.fc3 = SpectralLinear(hidden, out_dim, rank=min(rank, out_dim))
        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    @torch.no_grad()
    def retract_all(self):
        for layer in self.layers:
            layer.retract()

def test_xor():
    print("Training XOR Classification...")
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    Y = torch.tensor([[0.], [1.], [1.], [0.]])

    dense_model = DenseMLP(2, 32, 1)
    dense_opt = torch.optim.AdamW(dense_model.parameters(), lr=0.01)
    
    t0 = time.time()
    for _ in range(2000):
        pred = torch.sigmoid(dense_model(X))
        loss = F.binary_cross_entropy(pred, Y)
        dense_opt.zero_grad()
        loss.backward()
        dense_opt.step()
    dense_time = time.time() - t0
    dense_acc = ((torch.sigmoid(dense_model(X)) > 0.5).float() == Y).float().mean().item()

    sct_model = SpectralMLP(2, 32, 1, rank=16)
    sct_opt = torch.optim.AdamW(sct_model.parameters(), lr=0.01)
    
    t0 = time.time()
    for _ in range(2000):
        pred = torch.sigmoid(sct_model(X))
        loss = F.binary_cross_entropy(pred, Y)
        sct_opt.zero_grad()
        loss.backward()
        sct_opt.step()
        sct_model.retract_all()
    sct_time = time.time() - t0
    sct_acc = ((torch.sigmoid(sct_model(X)) > 0.5).float() == Y).float().mean().item()

    return {
        "dense": {"accuracy": dense_acc, "time_sec": round(dense_time, 3)},
        "sct": {"accuracy": sct_acc, "time_sec": round(sct_time, 3)}
    }

def test_sine():
    print("Training Sine Regression...")
    X = torch.linspace(-3.14, 3.14, 200).unsqueeze(1)
    Y = torch.sin(X)

    dense_model = DenseMLP(1, 64, 1)
    dense_opt = torch.optim.AdamW(dense_model.parameters(), lr=0.01)
    
    t0 = time.time()
    for _ in range(2000):
        pred = dense_model(X)
        loss = F.mse_loss(pred, Y)
        dense_opt.zero_grad()
        loss.backward()
        dense_opt.step()
    dense_time = time.time() - t0
    dense_loss = F.mse_loss(dense_model(X), Y).item()

    sct_model = SpectralMLP(1, 64, 1, rank=16)
    sct_opt = torch.optim.AdamW(sct_model.parameters(), lr=0.01)
    
    t0 = time.time()
    for _ in range(2000):
        pred = sct_model(X)
        loss = F.mse_loss(pred, Y)
        sct_opt.zero_grad()
        loss.backward()
        sct_opt.step()
        sct_model.retract_all()
    sct_time = time.time() - t0
    sct_loss = F.mse_loss(sct_model(X), Y).item()

    return {
        "dense": {"final_loss": round(dense_loss, 6), "time_sec": round(dense_time, 3)},
        "sct": {"final_loss": round(sct_loss, 6), "time_sec": round(sct_time, 3)}
    }

if __name__ == "__main__":
    results = {
        "xor_classification": test_xor(),
        "sine_regression": test_sine()
    }
    
    with open("mlp_proof_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Validation complete. Results saved to mlp_proof_results.json")