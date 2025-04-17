import torch
import torch.nn as nn

class FBNet(nn.Module):
    
    def __init__(self, num_classes=6):
        super(FBNet, self).__init__()

        # Convolutional Feature Extractor
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Input = binary silhouette (192×108×1)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (192x108 → 96x54) #Torch hat imer CxHxW
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (96x54 → 48x27)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (48x27 → 24x13)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (24x13 → 12x6)
        )

        # Global Average Pooling (Adjusting for 192x108)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 2))  # (12x6 → 1x2)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 512),  # Adjusted to match the new feature map size
            nn.ReLU(inplace=True),
            nn.Dropout(0.0), #before 0.4
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0), #before 0.3
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0), #before 0.2
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Ensures output is between [0,1]
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)  # Reduce to (batch_size, 256, 1, 2)
        x = self.fc(x)
        return x

# Create a model instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FBNet().to(device)

# Print model summary
try:
    from torchsummary import summary
    summary(model, (1, 192, 108))
except ImportError:
    print(model)
