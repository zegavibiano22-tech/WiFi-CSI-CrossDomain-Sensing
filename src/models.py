import torch
import torch.nn as nn

class BaselineCNNLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super(BaselineCNNLSTM, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        self.lstm = nn.LSTM(input_size=10*10, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = x.view(batch_size, -1, 10*10)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

class DANN(nn.Module):
    def __init__(self, num_classes=6):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        self.class_classifier = nn.Linear(16 * 10 * 10, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(16 * 10 * 10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_output = self.class_classifier(features)
        return class_output
