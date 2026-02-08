import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        hidden_dims = [32, 64, 128, 256, 512]
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(512 * 4 + 256, 512 * 4)
    
    def forward(self, x, condition):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, condition], dim=1)
        x = F.relu(self.fc(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.fc = nn.Linear(2 * latent_dim, 4 * 512)

        hidden_dims = [512, 256, 128, 64, 32]
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)
        result = self.fc(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class CVAE(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels):
        super(CVAE, self).__init__()
        self.projector_c = nn.Linear(256 * 64 * 2, latent_dim)
        
        self.projector_mu = nn.Linear(512 * 4, latent_dim)
        self.projector_logvar = nn.Linear(512 * 4, latent_dim)

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(latent_dim, out_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def initialize_training(self, train_args):
        # training parameters
        lr = train_args['lr']
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.batch_size = train_args['batch_size']

    def loss_function(self,
                      recon,
                      target,
                      mu,
                      log_var) -> dict:
        recons_loss =F.mse_loss(recon, target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return recons_loss, kld_loss

    def forward(self, 
                vt_torch, 
                encoded_image, 
                encoded_tactile, 
                recon_target = "image"):
        """
        vt_torch: "image" [batch_size, 12, 64, 64], "tactile1" [batch_size, 12, 32, 32]
        encoded_image: [batch_size, 256 * 64]
        encoded_tactile: [batch_size, 256 * 64]
        recon_target: "image", "tactile1", "both"
        """
        if recon_target == "image":
            source = vt_torch[recon_target][:, 9:, :, :]
        else:
            raise NotImplementedError
        condition = torch.cat((encoded_image, encoded_tactile), dim=1)
        condition = self.projector_c(condition)

        encoded_source = self.encoder(source, condition)
        mu = self.projector_mu(encoded_source)
        logvar = self.projector_logvar(encoded_source)
        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z, condition)

        return recon, mu, logvar
