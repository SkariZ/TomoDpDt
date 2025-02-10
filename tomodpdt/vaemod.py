import torch
import torch.nn as nn

import deeplay as dl

class ConvVAE(nn.Module):
    def __init__(
            self,
            input_shape,
            latent_dim,
            conv_channels=[64, 64, 64],
            dense_dim=128,
            activation='lrelu',
            output_activation='linear',
            dropout=0.0,
            ):
        super(ConvVAE, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.conv_channels = conv_channels
        self.dense_dim = dense_dim
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = dropout

        self.encoder = self.build_encoder()
        
        #self.fc_mu = nn.Linear(self.flattened_size(), latent_dim)
        #self.fc_log_var = nn.Linear(self.flattened_size(), latent_dim)
        #self.fc_decode = nn.Linear(latent_dim, self.flattened_size())
        self.flattened_size = self.get_flattened_size()
        self.H = self.calculate_H()
        self.decoder = self.build_decoder()

        self.fc_mu = dl.MultiLayerPerceptron(self.flattened_size, [32, 32], latent_dim)
        self.fc_var = dl.MultiLayerPerceptron(self.flattened_size, [32, 32], latent_dim)
        self.fc_dec = dl.MultiLayerPerceptron(latent_dim, [32, 32], self.flattened_size)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'lrelu':
            return nn.LeakyReLU(0.1)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'celu':
            return nn.CELU()
        else:
            return nn.Identity()

    def get_flattened_size(self):
        dummy_input = torch.zeros(1, *self.input_shape)  # Shape: (1, C, H, W)
        dummy_output = self.encoder(dummy_input)  # Shape: (1, C', H', W')
        return dummy_output.view(1, -1).shape[1]  # Flatten to (1, N) and return N
    
    def calculate_H(self):
        # Get input height and width
        input_height, input_width = self.input_shape[1], self.input_shape[2]
        
        # Number of pooling layers
        num_poolings = len(self.conv_channels)  # Based on the number of convolutional layers (with pooling)

        # Calculate reduced height and width after max pooling
        # Each max pooling with kernel size 2 reduces the size by a factor of 2
        reduced_height = input_height // (2 ** num_poolings)
        reduced_width = input_width // (2 ** num_poolings)
        
        return (reduced_height, reduced_width)

    def build_encoder(self):
        """
        Build the encoder.
        """
        encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], self.conv_channels[0], 3, 1, 1),
            #instancenorm2d 
            nn.InstanceNorm2d(self.conv_channels[0]),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.conv_channels[0], self.conv_channels[1], 3, 1, 1),
            nn.Conv2d(self.conv_channels[1], self.conv_channels[1], 3, 1, 1),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.conv_channels[1], self.conv_channels[2], 3, 1, 1),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.conv_channels[1], self.conv_channels[2], 3, 1, 1),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            )
        return encoder
    
    def build_decoder(self):
        """
        Build the decoder.
        """ 
        decoder = nn.Sequential(
            nn.Unflatten(1, (self.conv_channels[2], self.H[0], self.H[1])),
            nn.Conv2d(self.conv_channels[2], self.conv_channels[2], 3, 2, 1),
            nn.ConvTranspose2d(self.conv_channels[2], self.conv_channels[2], 3, 2, 1),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(self.conv_channels[2], self.conv_channels[2], 3, 2, 1),
            nn.ConvTranspose2d(self.conv_channels[2], self.conv_channels[1], 3, 2, 1),
            self.get_activation(self.activation),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(self.conv_channels[1], self.conv_channels[1], 3, 2, 1),
            nn.ConvTranspose2d(self.conv_channels[1], self.input_shape[0], 3, 2, 1),

            # Resize the output to the original size
            nn.Upsample(size=(self.input_shape[1], self.input_shape[2]), mode='bilinear'),

            nn.Conv2d(self.input_shape[0], self.input_shape[0], 3, 1, 1),
            self.get_activation(self.output_activation)
            )
        
        return decoder


class Dummy3d2d(nn.Module):
    def __init__(self):
        super(Dummy3d2d, self).__init__()

    def forward(self, x):
        #Return projection of the 3D volume
        return x.sum(dim=-1)


if __name__ == "__main__":

    N = 48

    #Test dummy 3D to 2D model
    dummy = Dummy3d2d()
    x = torch.randn(N, N, N)
    print(dummy(x).shape)

    vae = ConvVAE((2, N, N), latent_dim=2)

    x = torch.randn(8, 2, N, N)

    vae_model = dl.VariationalAutoEncoder(latent_dim=2, input_size=(N, N))
    vae_model.encoder=vae.encoder
    vae_model.decoder=vae.decoder
    vae_model.fc_mu=vae.fc_mu
    vae_model.fc_var=vae.fc_var
    vae_model.fc_dec=vae.fc_dec
    
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