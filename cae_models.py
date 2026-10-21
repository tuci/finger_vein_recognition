import torch.nn as nn
import torch.nn.functional as F

class CAE_patch(nn.Module):
    def __init__(self, num_layers, in_channels, disable_decoder=False, visualise=False):
        super(CAE_patch, self).__init__()
        self.disable_decoder = disable_decoder
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        self.in_features = 2 ** (num_layers + 3)
        self.visualise = visualise

        self.encoder = self.make_encoder(self.num_layers)
        self.decoder = self.make_decoder(self.num_layers)

        self.fcE = nn.Linear(in_features=self.in_features*1*1, out_features=32)
        self.fcD = nn.Linear(in_features=32, out_features=self.in_features*1*1)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1, stride=1)
        self.act = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self, input):
        out = self.encoder(input)
        out = out.view(-1, self.in_features*1*1)
        out = self.fcE(out)

        if self.disable_decoder:
            return out
        
        latent = out

        out = self.fcD(out)
        out = out.view(-1, self.in_features, 1, 1)
        out = self.decoder(out)

        out = self.conv(out)

        if self.visualise:
            return out, latent
        return out

    def make_encoder(self, num_layers):
        encoder_layers = []
        out_channels = self.in_channels
        for l in range(num_layers):
            if l == 0:
                padding = 0
                encoder_layers.append(self.conv_block(in_channels=1, out_channels=self.in_channels,
                                                      kernel_size=3, stride=2, padding=padding))
                in_channels = self.in_channels
            else:
                padding = 1
                out_channels = in_channels * 2
                encoder_layers.append(self.conv_block(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=3, stride=2, padding=padding))
                in_channels = out_channels
            self.out_channels = out_channels
        return nn.Sequential(*encoder_layers)

    def make_decoder(self, num_layers):
        decoder_layers = []
        in_channels = self.out_channels
        for l in range(num_layers):
            if l == 0:
                padding = 0
            else:
                padding = 1
            if l == num_layers - 1:
                out_channels = in_channels
            else:
                out_channels = in_channels // 2
            decoder_layers.append(self.deconv_block(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=3, stride=2, padding=padding))
            in_channels = out_channels

        return nn.Sequential(*decoder_layers)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU())
        return block

    def deconv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU())
        return block

    def init_weights(self, x):
        if isinstance(x, nn.Conv2d):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
        if isinstance(x, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
        if isinstance(x, nn.Linear):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
        if isinstance(x, nn.BatchNorm2d):
            nn.init.normal_(x.weight.data, 1.0, 0.02)
            nn.init.constant_(x.bias.data, 0)

