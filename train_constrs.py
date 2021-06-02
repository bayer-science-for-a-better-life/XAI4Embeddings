import argparse
import os
import torch
from scores.scores import noise_score, var_score
from scores.attributions import compute_importance_features
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.utilities.seed import seed_everything


def get_parser():
    parser = argparse.ArgumentParser(description='Script for training with XAI-derived scores as constraints.')

    parser.add_argument('--device', action='store', dest='device', default=6, type=int,
                        help="Which gpu device to use. Defaults to 6")

    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str,
                        help='Name of directory to save the results.')

    parser.add_argument('--objective', action='store', dest='objective', default='noise', type=str,
                        help='Which learning strategy to implement.')

    parser.add_argument('--emb_dim', action='store', dest='emb_dim', default=3, type=int,
                        help="Dimension of the bottleneck embedding. Defaults to 3")

    parser.add_argument('--seed', action='store', dest='seed', default=0, type=int,
                        help="Seed for reproducibility.")

    parser.add_argument('--coeff', action='store', dest='coeff', default=0.1, type=float,
                        help="Coefficient between losses. Defaults to 0.1")

    return parser.parse_args()


class Conv_NN(torch.nn.Module):
    def __init__(self, rep_dim: int):
        super(Conv_NN, self).__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False, track_running_stats=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.softplus(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.softplus(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MNIST_AE(pl.LightningModule):

    def __init__(self, objective: str = 'noise', rep_dim: int = 3, coeff: float = 0.1):
        super().__init__()

        self.objective = objective
        self.rep_dim = rep_dim
        self.coeff = coeff

        self.encoder_conv = Conv_NN(rep_dim=self.rep_dim)

        self.decoder = nn.Sequential(nn.Linear(rep_dim, 32), nn.ReLU(), nn.Linear(32, 28 * 28))
        self.classifier = nn.Linear(rep_dim, 10)

        self.compute_grads = True if self.objective in ['noise', 'var', 'only_noise', 'only_var', 'both',
                                                        'normal'] else False

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder_conv(x)

        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x.view(x.size(0), -1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.global_step % 20 == 0:

            imp_featu = compute_importance_features(pre_model=self.encoder_conv,
                                                    layer=self.encoder_conv.conv2,
                                                    data=x[[0]],
                                                    samples_idx=None,
                                                    n_samples=1,
                                                    device=self.device,
                                                    size=28,
                                                    with_grads=self.compute_grads)

            imp_noise = compute_importance_features(pre_model=self.encoder_conv,
                                                    layer=self.encoder_conv.conv2,
                                                    data=torch.rand(
                                                        size=(x.shape[0], x.shape[1], x.shape[2], x.shape[3])),
                                                    samples_idx=None,
                                                    n_samples=1,
                                                    device=self.device,
                                                    size=28,
                                                    with_grads=self.compute_grads)

            noise_sc = noise_score(imp_featu, imp_noise, transform='lin', grad=self.compute_grads).mean().to(
                self.device)
            var_sc = var_score(imp_featu,
                               transform='lin',
                               normalize_dim=True,
                               threshold=False,
                               grad=self.compute_grads).to(self.device).mean().to(self.device)
            self.log('noise_score', noise_sc, prog_bar=True, on_step=True, on_epoch=True)
            self.log('var_score', var_sc, prog_bar=True, on_step=True, on_epoch=True)
        else:
            noise_sc = 0.0
            var_sc = 0.0

        logits = F.log_softmax(self.classifier(z), dim=1)
        class_loss = F.nll_loss(logits, y)
        if self.objective == 'noise':
            tot_loss = class_loss - self.coeff * noise_sc + loss
        elif self.objective == 'only_noise':
            tot_loss = noise_sc
        elif self.objective == 'only_var':
            tot_loss = - var_sc
        elif self.objective == 'both':
            tot_loss = class_loss - self.coeff * noise_sc - self.coeff * var_sc + loss
        elif self.objective == 'var':
            tot_loss = class_loss - self.coeff * var_sc + loss
        elif self.objective == 'normal':
            tot_loss = class_loss + loss
        else:
            raise NotImplementedError
        self.log('class_loss', class_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('tot_loss', tot_loss, prog_bar=True, on_step=True, on_epoch=True)

        return tot_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder_conv(x)
        logits = F.log_softmax(self.classifier(z), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        return loss


if __name__ == '__main__':

    args = get_parser()
    seed_everything(args.seed)

    if args.emb_dim == 3:
        max_epochs = 15
        print(f'\nSetting n_epochs = {max_epochs}\n')
    elif args.emb_dim == 2 or args.emb_dim == 1:
        max_epochs = 20
        print(f'\nSetting n_epochs = {max_epochs}\n')
    elif args.emb_dim == 5:
        max_epochs = 10
        print(f'\nSetting n_epochs = {max_epochs}\n')
    elif args.emb_dim == 10:
        max_epochs = 5
        print(f'\nSetting n_epochs = {max_epochs}\n')
    else:
        max_epochs = 15

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    tb_logger = pl_loggers.TensorBoardLogger(f'logs/{args.save_dir}/')

    autoencoder = MNIST_AE(objective=args.objective, rep_dim=args.emb_dim, coeff=args.coeff)
    trainer = pl.Trainer(gpus=f'{args.device}', logger=tb_logger, log_every_n_steps=10, max_epochs=max_epochs)
    trainer.fit(autoencoder, DataLoader(train, batch_size=64), DataLoader(val, batch_size=64))

    print('\nScript concluded!')
