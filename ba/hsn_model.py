from torch import nn

class HSNet(nn.Module):

    def __init__(self, num_classes=6):
        super(HSNet, self).__init__()
        
        # Convolutional Feature Extractor
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=1, stride=1),  # Input = binary silhouette (192×108×1)
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # (192×108 → 192×108)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # (192×108 → 64×36)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, stride=3, padding=1), # (64×36 → 22×12)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1), # (22×12 → 22×12)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1), # (22×12 → 22×12)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.0),
            nn.Linear(128 * 12 * 22, 4096),  # Adjusted based on input size
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # Output size = 6
            nn.Sigmoid()  # Ensures output is between [0,1]
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        #out = out.view(out.size(0), -1)  # Flatten for FC layers
        out = self.fc(out)
        return out