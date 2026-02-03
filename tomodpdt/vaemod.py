import torch
import torch.nn as nn
import deeplay as dl


class ConvVAE(nn.Module):
    def __init__(
        self,
        input_shape,
        latent_dim=2,
        conv_channels=[64, 48, 32],
        dense_dim=128,
        activation="lrelu",
        output_activation="sigmoid",
        dropout=0.0,
    ):
        super().__init__()

        self.input_shape = input_shape  # (C,H,W)
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        self.dense_dim = dense_dim
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout

        if len(self.conv_channels) != 3:
            raise ValueError("conv_channels must have 3 elements, hardcoded for now...")

        # Safety: 3 pooling layers => H,W must be divisible by 8
        H, W = self.input_shape[1], self.input_shape[2]
        if (H % 8) != 0 or (W % 8) != 0:
            raise ValueError(f"ConvVAE expects H,W divisible by 8 (3 pools). Got H,W={(H, W)}")

        self.encoder = self.build_encoder()

        self.flattened_size = self.get_flattened_size()
        self.H = self.calculate_H()  # (H//8, W//8)
        self.decoder = self.build_decoder()

        self.fc_mu = dl.MultiLayerPerceptron(
            self.flattened_size,
            hidden_features=[16],
            out_features=latent_dim,
        )
        self.fc_var = dl.MultiLayerPerceptron(
            self.flattened_size,
            hidden_features=[16],
            out_features=latent_dim,
        )
        self.fc_dec = dl.MultiLayerPerceptron(
            latent_dim,
            hidden_features=[32],
            out_features=self.flattened_size,
        )

    def get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "lrelu":
            return nn.LeakyReLU(0.1)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "celu":
            return nn.CELU()
        elif activation == "linear":
            return nn.Identity()
        else:
            return nn.Identity()

    def get_flattened_size(self):
        dummy_input = torch.zeros(1, *self.input_shape)  # (1,C,H,W)
        dummy_output = self.encoder(dummy_input)         # (1,N)
        return dummy_output.view(1, -1).shape[1]

    def calculate_H(self):
        H, W = self.input_shape[1], self.input_shape[2]
        return (H // 8, W // 8)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
        )

    def upconv2_block(self, in_channels, out_channels, activation=True):
        # This EXACTLY doubles H,W with kernel=4,stride=2,pad=1
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        if activation:
            layers += [self.get_activation(self.activation)]
        layers += [nn.Dropout(self.dropout)]
        return nn.Sequential(*layers)

    def build_encoder(self):
        C = self.input_shape[0]
        c0, c1, c2 = self.conv_channels

        encoder = nn.Sequential(
            nn.Conv2d(C, c0, 3, 1, 1),
            nn.GroupNorm(num_groups=8, num_channels=c0),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),

            self.conv_block(c0, c0),
            nn.MaxPool2d(2),   # /2

            self.conv_block(c0, c1),
            nn.MaxPool2d(2),   # /4

            self.conv_block(c1, c2),
            nn.MaxPool2d(2),   # /8

            nn.Flatten(),
        )
        return encoder

    def build_decoder(self):
        C = self.input_shape[0]
        c0, c1, c2 = self.conv_channels
        h8, w8 = self.H  # H//8, W//8

        decoder = nn.Sequential(
            nn.Unflatten(1, (c2, h8, w8)),  # (c2, H/8, W/8)

            self.conv_block(c2, c2),

            self.upconv2_block(c2, c1),     # -> (c1, H/4, W/4)
            self.conv_block(c1, c1),

            self.upconv2_block(c1, c0),     # -> (c0, H/2, W/2)
            self.conv_block(c0, c0),

            self.upconv2_block(c0, C, activation=False),  # -> (C, H, W)

            nn.Conv2d(C, C, 3, 1, 1),
            self.get_activation(self.output_activation),
        )
        return decoder

if __name__ == "__main__":

    N = 256
    vae = ConvVAE((2, N, N), latent_dim=2)
    
    x = torch.randn(8, 2, N, N)

    vae_model = dl.VariationalAutoEncoder(latent_dim=2, input_size=(N, N))
    vae_model.encoder = vae.encoder
    vae_model.decoder = vae.decoder
    vae_model.fc_mu = vae.fc_mu
    vae_model.fc_var = vae.fc_var
    vae_model.fc_dec = vae.fc_dec
    
    vae_model.build()

    print(vae_model(x)[0].shape)

    #Print input / output shapes for all components
    print(vae_model.encoder(x).shape)
    print(vae_model.fc_mu(vae_model.encoder(x)).shape)
    print(vae_model.fc_var(vae_model.encoder(x)).shape)
    print(vae_model.fc_dec(vae_model.fc_mu(vae_model.encoder(x))).shape)

    #decoder should take the input of the fc_dec and return the output of the decoder
    print(vae_model.decoder(vae_model.fc_dec(vae_model.fc_mu(vae_model.encoder(x)))).shape)
    

    #Count the number of parameters
    print(sum(p.numel() for p in vae_model.parameters()))

    #Count the parameters in the encoder
    print("Encoder parameters:")
    print(sum(p.numel() for p in vae_model.encoder.parameters()))
    #Count the parameters in the decoder
    print("Decoder parameters:")
    print(sum(p.numel() for p in vae_model.decoder.parameters()))
    #Count the parameters in the fc_mu
    print("fc_mu parameters:")
    print(sum(p.numel() for p in vae_model.fc_mu.parameters()))
    #Count the parameters in the fc_var
    print("fc_var parameters:")
    print(sum(p.numel() for p in vae_model.fc_var.parameters()))
    #Count the parameters in the fc_dec
    print("fc_dec parameters:")
    print(sum(p.numel() for p in vae_model.fc_dec.parameters()))
          