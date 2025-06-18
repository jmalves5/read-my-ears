import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalActionLocalizationModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, num_classes=1, dropout=0.2):
        super(TemporalActionLocalizationModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)  # (B, T, D) => (B, T, H)
        self.fc = nn.Linear(hidden_dim, num_classes)  # => (B, T, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is float32
        x = x.to(torch.float32)

        # Apply LSTM
        output, _ = self.lstm(x)

        # Apply fully connected layer
        logits = self.fc(output)  # (B, T, 1)

        # Apply sigmoid activation to ensure outputs are in range [0,1]
        logits = self.sigmoid(logits)

        return logits  # (B, T, 1)
