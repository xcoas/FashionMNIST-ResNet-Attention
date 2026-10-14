import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Neural_FashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.res_block = ResidualBlock(64)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        self.flatten = nn.Flatten()
        
        self.linear_stack = nn.Sequential(
            nn.Linear(64 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.downsample(x)
        x = self.res_block(x)

        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        x_attn, attn_weights = self.multihead_attn(x_flat, x_flat, x_flat)
        x = x_attn.permute(0, 2, 1).contiguous().view(b, c, h, w)

        x = self.flatten(x)

        features = self.linear_stack[0](x)
        features = self.linear_stack[1](features)

        out = self.linear_stack[3](self.linear_stack[2](features))

        return out, attn_weights, features

model = Neural_FashionMNIST()