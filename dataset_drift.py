import argparse
import os
import torch
from scores.scores import noise_score
from scores.attributions import compute_importance_features
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='Script for Dataset Drift Detection.')

    parser.add_argument('--device', action='store', dest='device', default=6, type=int,
                        help="Which gpu device to use. Defaults to 6")

    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str,
                        help='Name of directory to save the results.')

    parser.add_argument('--objective', action='store', dest='objective', default='noise', type=str,
                        help='Which learning strategy.')

    parser.add_argument('--coeff', action='store', dest='coeff', default=0.2, type=float,
                        help='Noise controlling coefficient.')

    return parser.parse_args()


def normalize_simple(image):
    if image.max() > 1:
        image /= 255
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False, track_running_stats=False)
        self.conv3 = nn.Conv2d(4, 2, 3, bias=False, padding=2)
        # self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rep_dim = 32

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False, track_running_stats=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False, track_running_stats=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = F.interpolate(F.relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, objective: str = 'noise'):
        super().__init__()

        self.objective = objective

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.encoder = Encoder()

        # Decoder
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, x)  # x.view(x.size(0), -1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        val_loss = F.mse_loss(z, x)

        self.log('val_loss', val_loss, prog_bar=True)

        if self.objective == "noise":
            x_noise = x + 0.5 * torch.randn(x.shape).to(self.device)
            x_noise = torch.clip(x_noise, 0, 1)
            val_loss_noise = F.mse_loss(self(x_noise), x)
            self.log('val_loss_noise', val_loss_noise, prog_bar=True)

        return val_loss


if __name__ == '__main__':
    args = get_parser()

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    tb_logger = pl_loggers.TensorBoardLogger(f'logs/dataset_drift/')
    autoencoder = LitAutoEncoder()

    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    try:
        trainer = pl.Trainer(gpus=f'{args.device}', logger=tb_logger, max_epochs=10)
    except pl.utilities.exceptions.MisconfigurationException:
        trainer = pl.Trainer(logger=tb_logger, max_epochs=5)
    trainer.fit(autoencoder, DataLoader(train, batch_size=64), DataLoader(val, batch_size=64))

    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    autoencoder.eval()

    result = []

    for incr in range(10):
        coeff = incr * 0.1
        ns_x_conv2 = []
        ns_x_noise_conv2 = []
        count = 0
        for x, y in val:
            x_noise = x + coeff * torch.randn(x.shape)
            x_noise = torch.clip(x_noise, 0, 1)

            importance_layer_noise = compute_importance_features(pre_model=autoencoder.encoder,
                                                                 layer=autoencoder.encoder.conv2,
                                                                 data=torch.rand(1, 1, 28, 28),
                                                                 samples_idx=[0],
                                                                 n_samples=50,
                                                                 device=torch.device('cpu'),
                                                                 size=28,
                                                                 with_activations=False)
            importance_layer_drifted = compute_importance_features(pre_model=autoencoder.encoder,
                                                                   layer=autoencoder.encoder.conv2,
                                                                   data=x_noise.unsqueeze(dim=0),
                                                                   samples_idx=[0],
                                                                   n_samples=50,
                                                                   device=torch.device('cpu'),
                                                                   size=28,
                                                                   with_activations=False)
            importance_layer = compute_importance_features(pre_model=autoencoder.encoder,
                                                           layer=autoencoder.encoder.conv2,
                                                           data=x.unsqueeze(dim=0),
                                                           samples_idx=[0],
                                                           n_samples=50,
                                                           device=torch.device('cpu'),
                                                           size=28,
                                                           with_activations=False)

            importance_layer = [imp.squeeze() for imp in importance_layer]
            importance_layer_drifted = [imp.squeeze() for imp in importance_layer_drifted]
            importance_layer_noise = [imp.squeeze() for imp in importance_layer_noise]

            ns_x_conv2.append(noise_score(importance_layer, importance_layer_noise, summing='var'))
            ns_x_noise_conv2.append(noise_score(importance_layer_drifted, importance_layer_noise, summing='var'))

            count += 1

            print('Done with step', count)

            if count > 5:
                break

        result.append({
            'conv2_noise': np.array(ns_x_noise_conv2),
            'conv2': np.array(ns_x_conv2),
        })

    delta1 = []
    delta2 = []
    conv2_orig = []
    for sm in result:
        delta2.append((np.array(sm['conv2']) - np.array(sm['conv2_noise'])) / np.array(sm['conv2']))

    plt.boxplot(delta2, labels=("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"))
    plt.xlabel(r'Drift parameter $\lambda$', fontdict=None, labelpad=None)
    plt.ylabel('Relative Noise Score Difference', fontdict=None, labelpad=None)
    plt.show()
