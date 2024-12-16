from torch import nn

class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # Conv2d-1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) # Conv2d-3
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # MaxPool2d-5

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Conv2d-6
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # Conv2d-8
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # MaxPool2d-10

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Conv2d-11
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # Conv2d-13
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # MaxPool2d-15

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # Conv2d-16
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Conv2d-18
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # MaxPool2d-20

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Conv2d-21
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Conv2d-23

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # UpsamplingNearest2d-25
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)  # Conv2d-26
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # Conv2d-28

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # UpsamplingNearest2d-30
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)  # Conv2d-31
        self.conv14 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # Conv2d-33

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')  # UpsamplingNearest2d-35
        self.conv15 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)  # Conv2d-36
        self.conv16 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)  # Conv2d-38

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')  # UpsamplingNearest2d-40
        self.conv17 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)  # Conv2d-41
        self.conv18 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)  # Conv2d-43

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool2(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool3(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.maxpool4(x)

        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))

        # Decoder
        x = self.upsample1(x)
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))

        x = self.upsample2(x)
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))

        x = self.upsample3(x)
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))

        x = self.upsample4(x)
        x = self.relu(self.conv17(x))
        x = self.conv18(x)

        return x