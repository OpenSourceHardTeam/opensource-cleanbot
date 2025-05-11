import torch
from encoder.jamo_encoder import VECTOR_SIZE
from torch import nn

# model_v3
class ProfanityCNN(nn.Module):
    def __init__(self, input_dim=VECTOR_SIZE, seq_len=600):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=6)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=9)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=12)
        self.pool3 = nn.MaxPool1d(kernel_size=3)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, seq_len)
            x = self.pool1(torch.relu(self.conv1(dummy_input)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = self.pool3(torch.relu(self.conv3(x)))
            self.flatten_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        return self.fc3(x).squeeze()

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProfanityCNN(input_dim=VECTOR_SIZE).to(device)
model.load_state_dict(torch.load("model/clean_bot_model_v3.pt", map_location=device))
model.eval()
