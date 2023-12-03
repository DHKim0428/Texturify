### mine ###
import os
import copy
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import warnings
warnings.filterwarnings(action='ignore')
############

from pathlib import Path

import torch
from cleanfid import fid
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import GraphDataLoader, to_device, to_vertex_colors_scatter
from model.differentiable_renderer import DifferentiableRenderer
from model.graph_generator_u_deep import Generator
from model.graph import TwinGraphEncoder, TwinGraphFeature, TwinGraphFeatureLatent
from dataset.mesh_real_features_combined import FaceGraphMeshDataset as CombinedData
from dataset.mesh_real_features import FaceGraphMeshDataset as ChairData
from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset as CarData
from torch_ema import ExponentialMovingAverage
import hydra
from pathlib import Path

from util.misc import get_parameters_from_state_dict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


def render_faces(R, face_colors, batch, render_size, image_size):
    rendered_color = R.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), resolution=render_size)
    ret_val = rendered_color.permute((0, 3, 1, 2))
    if render_size != image_size:
        ret_val = torch.nn.functional.interpolate(ret_val, (image_size, image_size), mode='bilinear', align_corners=True)
    return ret_val

class SampleEval:
    def __init__(self, config, checkpoints=None, options="chair", max_iter=20):
        self.output_dir = Path("./output")
        self.output_dir_ours = self.output_dir / "result_compare_{0}".format(options)
        self.output_dirs = {
            'baseline': self.output_dir_ours / "baseline",
            'sum': self.output_dir_ours / "sum",
            'sum_extend': self.output_dir_ours / "sum_extend",
            'latent': self.output_dir_ours / "latent",
            'discriminator': self.output_dir_ours / "discriminator",
        }
        self.output_dir_ours.mkdir(exist_ok=True, parents=True)
        for _, dir in self.output_dirs.items():
            dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device("cuda")
        config.batch_size = 4
        config.image_size = 256
        config.render_size = 512
        self.num_latent = 4
        self.max_iter = max_iter
        if options == "chair":
            self.eval_dataset = ChairData(config)
        elif options == "car":
            self.eval_dataset = CarData(config)
        else:
            return
        self.eval_loader = GraphDataLoader(self.eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)
        self.R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
        self.config = config
        self.checkpoints = checkpoints
        
        self.load_baseline()
        self.load_sum()
        self.load_sum_extend()
        self.load_latent()
        self.load_discriminator()

    def load_baseline(self):
        config = self.config
        device = self.device
        self.E_base = TwinGraphEncoder(self.eval_dataset.num_feats, 1).to(device)
        self.G_base = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E_base.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(self.checkpoints['baseline'][0], map_location=device)["state_dict"]
        self.G_base.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = ExponentialMovingAverage(self.G_base.parameters(), 0.995)
        ema.load_state_dict(torch.load(self.checkpoints['baseline'][1], map_location=device))
        ema.copy_to([p for p in self.G_base.parameters() if p.requires_grad])
        self.E_base.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        self.G_base.eval()
        self.E_base.eval()
    
    def load_sum(self):
        config = self.config
        device = self.device
        self.E_sum = TwinGraphEncoder(self.eval_dataset.num_feats, 1).to(device)
        self.F_sum = TwinGraphFeature(self.eval_dataset.num_feats, 1, batch_size=config.batch_size).to(device)
        self.G_sum = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E_sum.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(self.checkpoints['sum'][0], map_location=device)["state_dict"]
        self.G_sum.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = ExponentialMovingAverage(self.G_sum.parameters(), 0.995)
        ema.load_state_dict(torch.load(self.checkpoints['sum'][1], map_location=device))
        ema.copy_to([p for p in self.G_sum.parameters() if p.requires_grad])
        self.E_sum.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        f_dict = torch.load("./runs/08111238_StyleGAN23D-Feature_fast_dev/checkpoints/_epoch=14.ckpt", map_location=device)["state_dict"]
        self.F_sum.load_state_dict(get_parameters_from_state_dict(f_dict, "F"))

        self.E_sum.eval()
        self.G_sum.eval()
        self.F_sum.eval()
    
    def load_sum_extend(self):
        config = self.config
        device = self.device
        self.E_sum_extend = TwinGraphEncoder(self.eval_dataset.num_feats, 1).to(device)
        self.G_sum_extend = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E_sum_extend.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(self.checkpoints['sum_extend'][0], map_location=device)["state_dict"]
        self.G_sum_extend.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = ExponentialMovingAverage(self.G_sum_extend.parameters(), 0.995)
        ema.load_state_dict(torch.load(self.checkpoints['sum_extend'][1], map_location=device))
        ema.copy_to([p for p in self.G_sum_extend.parameters() if p.requires_grad])
        self.E_sum_extend.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        self.E_sum_extend.eval()
        self.G_sum_extend.eval()
    
    def load_latent(self):
        config = self.config
        device = self.device
        self.E_latent = TwinGraphEncoder(self.eval_dataset.num_feats, 1).to(device)
        self.F_latent = TwinGraphFeatureLatent(self.eval_dataset.num_feats, 1, batch_size=config.batch_size).to(device)
        self.G_latent = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E_latent.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(self.checkpoints['latent'][0], map_location=device)["state_dict"]
        self.G_latent.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = ExponentialMovingAverage(self.G_latent.parameters(), 0.995)
        ema.load_state_dict(torch.load(self.checkpoints['latent'][1], map_location=device))
        ema.copy_to([p for p in self.G_latent.parameters() if p.requires_grad])
        self.E_latent.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
        f_dict = torch.load("./runs/09111258_StyleGAN23D-Feature-Latent_fast_dev/checkpoints/_epoch=39.ckpt", map_location=device)["state_dict"]
        self.F_latent.load_state_dict(get_parameters_from_state_dict(f_dict, "F"))

        self.E_latent.eval()
        self.G_latent.eval()
        self.F_latent.eval()
    
    def load_discriminator(self):
        config = self.config
        device = self.device
        self.E_discrim = TwinGraphEncoder(self.eval_dataset.num_feats, 1).to(device)
        self.G_discrim = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=self.E_discrim.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
        state_dict = torch.load(self.checkpoints['discriminator'][0], map_location=device)["state_dict"]
        self.G_discrim.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
        ema = ExponentialMovingAverage(self.G_discrim.parameters(), 0.995)
        ema.load_state_dict(torch.load(self.checkpoints['discriminator'][1], map_location=device))
        ema.copy_to([p for p in self.G_discrim.parameters() if p.requires_grad])
        self.E_discrim.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))

        self.G_discrim.eval()
        self.E_discrim.eval()

    def eval_baseline(self, iter_idx, eval_batch):
        config = self.config
        device = self.device
        with torch.no_grad():
            shape = self.E_base(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            for z_idx in range(self.num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = self.G_base(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(self.R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], self.output_dirs['baseline'] / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    def eval_sum(self, iter_idx, eval_batch):
        config = self.config
        device = self.device
        with torch.no_grad():
            shape = self.E_sum(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            feature, _ = self.F_sum(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            shape[-1] += feature[-1]

            for z_idx in range(self.num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = self.G_sum(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(self.R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], self.output_dirs['sum'] / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    def eval_sum_extend(self, iter_idx, eval_batch):
        config = self.config
        device = self.device
        with torch.no_grad():
            shape = self.E_sum_extend(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            feature, _ = self.F_sum(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            shape = [c+f for c, f in zip(shape, feature)]

            for z_idx in range(self.num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = self.G_sum_extend(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(self.R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], self.output_dirs['sum_extend'] / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    def eval_latent(self, iter_idx, eval_batch):
        config = self.config
        device = self.device
        with torch.no_grad():
            shape = self.E_latent(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            feature, _ = self.F_latent(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])

            for z_idx in range(self.num_latent):
                z = torch.randn(config.batch_size, config.latent_dim // 2).to(device)
                z = torch.cat((z, feature), dim=1)

                fake = self.G_latent(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(self.R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], self.output_dirs['latent'] / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    def eval_discriminator(self, iter_idx, eval_batch):
        config = self.config
        device = self.device
        with torch.no_grad():
            shape = self.E_discrim(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            for z_idx in range(self.num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = self.G_discrim(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(self.R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], self.output_dirs['discriminator'] / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    def run(self):
        for iter_idx, batch in enumerate(tqdm(self.eval_loader)):
            if iter_idx >= self.max_iter:
                return
            eval_batch = to_device(batch, self.device)
            self.eval_baseline(iter_idx, eval_batch) # 99, 000181485
            self.eval_sum(iter_idx, eval_batch)
            self.eval_sum_extend(iter_idx, eval_batch)
            self.eval_latent(iter_idx, eval_batch)
            self.eval_discriminator(iter_idx, eval_batch)


@hydra.main(config_path='../config', config_name='stylegan2', version_base=None)
def main(config):
    baseline_path = Path('./runs/_28101158_StyleGAN23D-Combined_fast_dev/checkpoints')
    sum_path = Path('./runs/_20111013_StyleGAN23D-Combined-Feature-sum_fast_dev/checkpoints')
    sum_extend_path = Path('./runs/_12110622_StyleGAN23D-Combined-Feature-sum-extended_fast_dev/checkpoints')
    latent_path = Path('./runs/_12110616_StyleGAN23D-Combined-Feature-latent_fast_dev/checkpoints')
    discrim_path = Path('./runs/22111714_StyleGAN23D-Combined-Discriminator_fast_dev/checkpoints')

    checkpoints = {
        'baseline': [baseline_path / "_epoch=99.ckpt", baseline_path / "ema_000181485.pth"],
        'sum': [sum_path / "_epoch=69.ckpt", sum_path / "ema_000127040.pth"],
        'latent': [latent_path / "_epoch=49.ckpt", latent_path / "ema_000090743.pth"],
        'sum_extend': [sum_extend_path / "_epoch=49.ckpt", sum_extend_path / "ema_000090743.pth"],
        'discriminator': [discrim_path / "_epoch=69.ckpt", discrim_path / "ema_000127040.pth"],
    }
    evalrunner = SampleEval(config, checkpoints, "chair", max_iter=100)
    evalrunner.run()

if __name__ == "__main__":
    main()


