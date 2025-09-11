# -*- coding:utf-8 -*-
import os
import json
import sys
import mne
import torch
import random
import shutil
import argparse
import warnings
import numpy as np
import torch.optim as opt
from ..model.utils import model_size, set_random_seed
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from ..dataset.utils import split_train_test_val_files
from data_loader import TorchDataset
from ..model.neuronet import NeuroNet
warnings.filterwarnings(action='ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.n_fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config
        self.model_name = config['name']
        self.ds_cfg = config['Dataset']

        self.fs = self.ds_cfg['rfreq']
        self.raw_fs = self.ds_cfg['sfreq']
        self.second = self.ds_cfg['second']
        self.time_window = self.ds_cfg['time_window']
        self.time_step = self.ds_cfg['time_step']
        self.data_scaler = self.ds_cfg['data_scaler']
        self.base_path = self.ds_cfg['base_path']
        self.ckpt_path = self.ds_cfg['ckpt_path']
        self.k_splits = self.ds_cfg['k_splits']

        self.encoder_embed_dim = self.config['Encoder']['encoder_embed_dim']
        self.encoder_heads = self.config['Encoder']['encoder_heads']
        self.encoder_depths = self.config['Encoder']['encoder_depths']
        self.alpha = self.config['Encoder']['alpha']

        self.decoder_embed_dim = self.config['Decoder']['decoder_embed_dim']
        self.decoder_heads = self.config['Decoder']['decoder_heads']
        self.decoder_depths = self.config['Decoder']['decoder_depths']

        self.projection_hidden = self.config['Encoder']['projection_hidden']

        self.temperature = self.config['training_params']['temperature']
        self.mask_ratio = self.config['training_params']['mask_ratio']
        self.print_point = self.config['training_params']['print_point']

        self.train_epochs = self.config['training_params']['train_epochs']
        self.warmup_epochs = self.config['training_params']['train_warmup_epoch']

        self.batch_size = self.config['training_params']['batch_size']
        self.train_batch_accumulation = self.config['training_params']['train_batch_accumulation']
        self.eff_batch_size = self.batch_size * self.train_batch_accumulation

        self.train_base_learning_rate = self.config['training_params']['base_learning_rate']

        self.lr = self.train_base_learning_rate * self.eff_batch_size / 256
        self.model = self.build_model()

        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.train_paths, self.val_paths, self.eval_paths = self.data_paths()
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_epochs)
        self.tensorboard_path = os.path.join(self.ckpt_path, self.model_name,
                                             str(self.n_fold), 'tensorboard')

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('Model Size : {0:.2f}MB'.format(model_size(self.model)))

        print('Frame Size : {}'.format(self.model.num_patches))
        print('Leaning Rate : {0}'.format(self.lr))
        print('Validation Paths : {0}'.format(len(self.val_paths)))
        print('Evaluation Paths : {0}'.format(len(self.eval_paths)))

    def build_model(self):
        model = NeuroNet(
            fs=self.fs, second=self.second, time_window=self.time_window, time_step=self.time_step,
            encoder_embed_dim=self.encoder_embed_dim, encoder_heads=self.encoder_heads,
            encoder_depths=self.encoder_depths, decoder_embed_dim=self.decoder_embed_dim,
            decoder_heads=self.decoder_heads, decoder_depths=self.decoder_depths,
            projection_hidden=self.projection_hidden, temperature=self.temperature
        )
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def train(self):
        print('K-Fold : {}/{}'.format(self.n_fold , self.k_splits))
        train_dataset = TorchDataset(paths=self.train_paths, sfreq=self.raw_fs, rfreq=self.fs,
                                     scaler=self.data_scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = TorchDataset(paths=self.val_paths, sfreq=self.raw_fs, rfreq=self.fs,
                                   scaler=self.data_scaler)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True)
        eval_dataset = TorchDataset(paths=self.eval_paths, sfreq=self.raw_fs, rfreq=self.fs,
                                    scaler=self.data_scaler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, drop_last=True)

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_dataloader:
                x = x.to(device)
                out = self.model(x, mask_ratio=self.mask_ratio)
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = out

                loss = recon_loss + self.alpha * contrastive_loss
                loss.backward()

                if (step + 1) % self.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                          '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                          '[Total Loss] : {4:02.4f}  [Contrastive Acc] : {5:02.4f}'.format(
                            epoch, total_step + 1, recon_loss, contrastive_loss, loss,
                            self.compute_metrics(cl_logits, cl_labels)))

                self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                self.tensorboard_writer.add_scalar('Contrastive loss', contrastive_loss, total_step)
                self.tensorboard_writer.add_scalar('Total loss', loss, total_step)

                step += 1
                total_step += 1

            val_acc, val_mf1 = self.linear_probing(val_dataloader, eval_dataloader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                epoch, val_acc * 100, val_mf1 * 100))
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)

            self.optimizer.step()
            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

    def linear_probing(self, val_dataloader, eval_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(val_dataloader), \
                                               self.get_latent_vector(eval_dataloader)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

        model = KNeighborsClassifier()
        model.fit(train_x, train_y)

        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                latent = self.model.forward_latent(x)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)
        return total_x, total_y

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(self.ckpt_path, self.model_name, str(self.n_fold), 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save({
            'model_name': 'NeuroNet',
            'model_state': model_state,
            'model_parameter': {
                'fs': self.fs, 'second': self.second,
                'time_window': self.time_window, 'time_step': self.time_step,
                'encoder_embed_dim': self.encoder_embed_dim, 'encoder_heads': self.encoder_heads,
                'encoder_depths': self.encoder_depths,
                'decoder_embed_dim': self.decoder_embed_dim, 'decoder_heads': self.decoder_heads,
                'decoder_depths': self.decoder_depths,
                'projection_hidden': self.projection_hidden, 'temperature': self.temperature
            },
            'hyperparameter': self.__dict__,
            'paths': {'train_paths': self.train_paths, 'ft_paths': self.val_paths, 'eval_paths': self.eval_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))

    def data_paths(self):
        kf = split_train_test_val_files(base_path=self.base_path, n_splits=self.k_splits)

        paths = kf[self.n_fold]
        train_paths, ft_paths, eval_paths = paths['train_paths'], paths['ft_paths'], paths['eval_paths']
        return train_paths, ft_paths, eval_paths

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    for fold in range(1, config['Dataset']['k_splits'] + 1):
        trainer = Trainer(args, fold, config)
        trainer.train()


if __name__ == '__main__':
    main()
