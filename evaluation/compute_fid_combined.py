### mine ###
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import warnings
warnings.filterwarnings(action='ignore')
############

import torch
from cleanfid import fid
from pytorch_fid import fid_score as fid_2
from dataset.mesh_real_features_combined import FaceGraphMeshDataset
from dataset import GraphDataLoader, to_device
from torchvision.utils import save_image
import hydra
from pathlib import Path
from tqdm import tqdm 

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


@hydra.main(config_path='../config', config_name='stylegan2_combined')
def validation_epoch_end(config):
    # category = "chair"
    # if category == "chair":
    #     odir_fake = Path("./output/27101203_2_1")
    #     odir_real = Path("./output/chair_real")
    #     config.image_size = 256
    # else:
    #     odir_fake = Path("./output/ours_vis_interesting")
    #     odir_real = Path("./output/cars_real")
    #     config.image_size = 512
    odir_real = Path("./output/chair_real")
    odir_fake = Path("./output/28101158_chairs")
    
    odir_real.mkdir(exist_ok=True, parents=True)

    device = "cuda"
    # val_num = len(os.listdir(str(odir_fake)))
    # val_dataset = FaceGraphMeshDataset(config, val_num)
    # val_loader = GraphDataLoader(val_dataset, config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers)
    # for iter_idx, batch in enumerate(tqdm(val_loader, desc="real_image_save")):
    #     batch = to_device(batch, device)
    #     real_render = batch['real'].cpu()
    #     for batch_idx in range(real_render.shape[0]):
    #         save_image(real_render[batch_idx], odir_real / f"{iter_idx}_{batch_idx}.jpg", value_range=(-1, 1), normalize=True)

    fid_score = fid.compute_fid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    print(f'FID: {fid_score:.3f}')
    kid_score = fid.compute_kid(str(odir_real), str(odir_fake), device="cuda", num_workers=0)
    print(f'KID: {kid_score:.3f}')

if __name__=="__main__":
    validation_epoch_end()