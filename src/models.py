import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, features: list[int] | None = None, num_classes: int = 1):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]
        assert in_channels > 0
        assert len(features) > 0
        assert num_classes > 0

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features[:-1]:
            self.encoder.append(DoubleConvBlock(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        self.bottleneck = DoubleConvBlock(in_channels=in_channels, out_channels=features[-1])
        in_channels = features[-1]

        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            in_channels = feature
            self.decoder.append(DoubleConvBlock(in_channels=in_channels * 2, out_channels=feature))

        self.final_conv = nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for enc_block in self.encoder:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            upsample = self.decoder[idx]
            conv_block = self.decoder[idx + 1]

            x = upsample(x)
            skip = skip_connections[idx // 2]

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(input=x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = conv_block(x)

        return self.final_conv(x)
