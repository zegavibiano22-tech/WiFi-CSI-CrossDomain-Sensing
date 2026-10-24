import torch.nn as nn
from torch.autograd import Function


class CNNLSTMEncoder(nn.Module):
    """Input (B,1,T,H,W) -> fitur (B,64). Dipakai bersama oleh baseline dan DANN."""
    def __init__(self, feat_dim=64, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.bridge = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                    nn.Linear(64 * 5 * 5, 128), nn.ReLU())
        self.lstm = nn.LSTM(128, feat_dim, batch_first=True)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        f = self.bridge(self.cnn(x)).reshape(b, t, -1)
        out, _ = self.lstm(f)
        return out[:, -1]


class BaselineCNNLSTM(nn.Module):
    def __init__(self, num_classes=6, feat_dim=64):
        super().__init__()
        self.encoder = CNNLSTMEncoder(feat_dim)
        self.cls = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.cls(self.encoder(x))

    def extract(self, x):
        return self.encoder(x)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class DANN(nn.Module):
    def __init__(self, num_classes=6, num_domains=2, feat_dim=64):
        super().__init__()
        self.encoder = CNNLSTMEncoder(feat_dim)
        self.cls = nn.Linear(feat_dim, num_classes)
        self.dom = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Linear(32, num_domains))

    def forward(self, x, alpha=1.0):
        f = self.encoder(x)
        return self.cls(f), self.dom(GradReverse.apply(f, alpha))

    def extract(self, x):
        return self.encoder(x)
