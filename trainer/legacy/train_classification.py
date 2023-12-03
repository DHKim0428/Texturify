### mine ###
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')
############

import torch
import hydra
import torchmetrics
# import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from dataset.mesh_real_features_combined import FaceGraphMeshDataset
from dataset import GraphDataLoader
from torch_ema import ExponentialMovingAverage
from model.graph import TwinGraphEncoder, TwinGraphFeature, TwinGraphFeatureLatent
import pytorch_lightning as pl
from trainer import create_trainer
from util.timer import Timer

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

class StyleGAN2Trainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.train_set = FaceGraphMeshDataset(config)
        self.val_set = FaceGraphMeshDataset(config, config.num_eval_images)
        self.F = TwinGraphFeatureLatent(self.train_set.num_feats, 1, batch_size=config.batch_size)
        self.automatic_optimization = False
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)

    def configure_optimizers(self):
        f_opt = torch.optim.Adam([
            {'params': list(self.F.parameters()), 'lr': self.config.lr_e, 'eps': 1e-8, 'weight_decay': 1e-4}
        ])
        return f_opt
    
    def get_feature(self, batch, limit_batch_size=False):
        # print(batch['x'].shape)
        return self.F(batch['x'], batch['graph_data']['ff2_maps'][0], batch['graph_data'])
        
    def feature_step(self, batch):
        # print(batch['category'])
        label = torch.tensor([(1 if b == 'car' else 0) for b in batch['category']])
        label = label.to(self.device)
        f_opt = self.optimizers()
        f_opt.zero_grad(set_to_none=True)
        _, output = self.get_feature(batch)
        feature_loss = torch.nn.functional.cross_entropy(output, label)
        self.manual_backward(feature_loss)
        log_feature_loss = feature_loss.item()
        step(f_opt, self.F)
        acc = self.train_acc(output, label)
        self.log("F", log_feature_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        label = torch.tensor([(1 if b == 'car' else 0) for b in batch['category']])
        label = label.to(self.device)
        _, output = self.get_feature(batch)
        feature_loss = torch.nn.functional.cross_entropy(output, label)
        log_feature_loss = feature_loss.item()
        acc = self.train_acc(output, label)
        # self.log("val_F", log_feature_loss, on_step=True, on_epoch=False)
        # self.log("val_acc", acc, on_step=True, on_epoch=False)
        return {'val_F': log_feature_loss, 'val_acc': acc}


    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        avg_loss = torch.tensor([x['val_F'] for x in _val_step_outputs]).mean()
        avg_acc = torch.tensor([x['val_acc'] for x in _val_step_outputs]).mean()
        print("avg_loss: {0:.3f}, avg_acc: {1:.3f}".format(avg_loss, avg_acc))
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True)
        self.log("val_acc", avg_acc, on_step=False, on_epoch=True)

    def train_dataloader(self):
        return GraphDataLoader(self.train_set, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return GraphDataLoader(self.val_set, self.config.batch_size, shuffle=True, drop_last=True, num_workers=self.config.num_workers)

    def training_step(self, batch, batch_idx):
        self.feature_step(batch)
    


def step(opt, module):
    for param in module.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
    opt.step()


@hydra.main(config_path='../config', config_name='stylegan2_combined')
def main(config):
    trainer = create_trainer("StyleGAN23D-Feature-Latent", config)
    model = StyleGAN2Trainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
