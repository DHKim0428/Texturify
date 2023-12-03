### mine ###
import os
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
import hydra
from pathlib import Path

from util.misc import get_parameters_from_state_dict

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


OUTPUT_DIR = Path("./output")
REAL_DIR = Path("/cluster_HDD/gondor/ysiddiqui/surface_gan_eval/photoshape/real")
EXP_NAME = "fast_dev"

CHECKPOINT = "./checkpoints/D_gen/best.ckpt"
CHECKPOINT_EMA = "./checkpoints/D_gen/ema_best.pth"

def render_faces(R, face_colors, batch, render_size, image_size):
    rendered_color = R.render(batch['vertices'], batch['indices'], to_vertex_colors_scatter(face_colors, batch), batch["ranges"].cpu(), resolution=render_size)
    ret_val = rendered_color.permute((0, 3, 1, 2))
    if render_size != image_size:
        ret_val = torch.nn.functional.interpolate(ret_val, (image_size, image_size), mode='bilinear', align_corners=True)
    return ret_val


# Ours
@hydra.main(config_path='../config', config_name='stylegan2_combined', version_base=None)
def evaluate_our_gan(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder, TwinGraphFeature
    from dataset.mesh_real_features_combined import FaceGraphMeshDataset
    from torch_ema import ExponentialMovingAverage
    config.batch_size = 2
    # config.views_per_sample = 4
    config.image_size = 256
    # config.image_size = 512
    config.render_size = 512
    # config.num_mapping_layers = 5
    # config.g_channel_base = 32768
    # config.g_channel_max = 768
    # config.dataset_path = "../data/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color"
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"{EXP_NAME}_{config.views_per_sample}_{num_latent}"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = ExponentialMovingAverage(G.parameters(), 0.995)
    ema.load_state_dict(torch.load(CHECKPOINT_EMA, map_location=device))
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()
    
    with torch.no_grad():
        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    odir_real = Path("./output/combined")
    odir_fake = OUTPUT_DIR_OURS
    fid_score = fid.compute_fid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    kid_score = fid.compute_kid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    file_name = './combined_discriminator_{}.txt'.format(EXP_NAME)
    with open(file_name, 'a+') as file:
        file.write(f'FID: {fid_score:.4f}, KID: {kid_score:.4f}\n')

    return 


# Ours_car
@hydra.main(config_path='../config', config_name='stylegan2_car', version_base=None)
def evaluate_our_gan_car(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.meshcar_real_features_ff2 import FaceGraphMeshDataset
    from torch_ema import ExponentialMovingAverage
    config.batch_size = 2
    config.views_per_sample = 2
    config.image_size = 256
    config.render_size = 512
    # config.num_mapping_layers = 8
    config.g_channel_base = 32768
    config.g_channel_max = 512
    num_latent = 4

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"{EXP_NAME}_{config.views_per_sample}_{num_latent}_car"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = ExponentialMovingAverage(G.parameters(), 0.995)
    ema.load_state_dict(torch.load(CHECKPOINT_EMA, map_location=device))
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()
    
    with torch.no_grad():
        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    odir_real = Path("./output/cars_real")
    odir_fake = OUTPUT_DIR_OURS
    fid_score = fid.compute_fid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    kid_score = fid.compute_kid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    file_name = './combined_discriminator_car_{}.txt'.format(EXP_NAME)
    with open(file_name, 'a+') as file:
        file.write(f'FID: {fid_score:.4f}, KID: {kid_score:.4f}\n')

    return 


# Ours_chair
@hydra.main(config_path='../config', config_name='stylegan2', version_base=None)
def evaluate_our_gan_chair(config):
    from model.graph_generator_u_deep import Generator
    from model.graph import TwinGraphEncoder
    from dataset.mesh_real_features import FaceGraphMeshDataset   
    from torch_ema import ExponentialMovingAverage
    config.batch_size = 2
    # config.views_per_sample = 4
    config.image_size = 256
    # config.image_size = 512
    config.render_size = 512
    # config.num_mapping_layers = 5
    # config.g_channel_base = 32768
    # config.g_channel_max = 768
    num_latent = 1

    OUTPUT_DIR_OURS = OUTPUT_DIR / f"{EXP_NAME}_{config.views_per_sample}_{num_latent}_chair"
    OUTPUT_DIR_OURS.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda")
    eval_dataset = FaceGraphMeshDataset(config)
    eval_loader = GraphDataLoader(eval_dataset, config.batch_size, drop_last=True, num_workers=config.num_workers)

    E = TwinGraphEncoder(eval_dataset.num_feats, 1).to(device)
    G = Generator(config.latent_dim, config.latent_dim, config.num_mapping_layers, config.num_faces, 3, e_layer_dims=E.layer_dims, channel_base=config.g_channel_base, channel_max=config.g_channel_max).to(device)
    R = DifferentiableRenderer(config.render_size, "bounds", config.colorspace)
    state_dict = torch.load(CHECKPOINT, map_location=device)["state_dict"]
    G.load_state_dict(get_parameters_from_state_dict(state_dict, "G"))
    ema = ExponentialMovingAverage(G.parameters(), 0.995)
    ema.load_state_dict(torch.load(CHECKPOINT_EMA, map_location=device))
    ema.copy_to([p for p in G.parameters() if p.requires_grad])
    E.load_state_dict(get_parameters_from_state_dict(state_dict, "E"))
    G.eval()
    E.eval()

    with torch.no_grad():
        for iter_idx, batch in enumerate(tqdm(eval_loader)):
            eval_batch = to_device(batch, device)
            shape = E(eval_batch['x'], eval_batch['graph_data']['ff2_maps'][0], eval_batch['graph_data'])
            for z_idx in range(num_latent):
                z = torch.randn(config.batch_size, config.latent_dim).to(device)
                fake = G(eval_batch['graph_data'], z, shape, noise_mode='const')
                fake_render = render_faces(R, fake, eval_batch, config.render_size, config.image_size)
                for batch_idx in range(fake_render.shape[0]):
                    save_image(fake_render[batch_idx], OUTPUT_DIR_OURS / f"{iter_idx}_{batch_idx}_{z_idx}.jpg", value_range=(-1, 1), normalize=True)

    odir_real = Path("./output/chair_real")
    odir_fake = OUTPUT_DIR_OURS
    fid_score = fid.compute_fid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    kid_score = fid.compute_kid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    file_name = './combined_discriminator_chair_{}.txt'.format(EXP_NAME)
    with open(file_name, 'a+') as file:
        file.write(f'FID: {fid_score:.4f}, KID: {kid_score:.4f}\n')

    return 


def main():
    evaluate_our_gan()
    evaluate_our_gan_car()
    evaluate_our_gan_chair()
        


if __name__ == "__main__":
    main()

