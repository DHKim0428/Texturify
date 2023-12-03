# Texturify: Generating Textures on 3D Shape Surfaces

## About

This is the academic project of [KAIST CS479 course](https://mhsung.github.io/kaist-cs479-fall-2023/) in 2023 Fall semester.

## Dependencies

It is recommended to create a new conda environment:

```commandline
conda create --name texturify python=3.9
conda activate texturify
```

Install python requirements:

```commandline
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Install trimesh from the authors' fork:
```bash
cd ~
git clone git@github.com:nihalsid/trimesh.git
cd trimesh
python setup.py install
```

Also, for differentiable rendering we use `nvdiffrast`. You'll need to install its dependencies:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl
```

Install `nvdiffrast` from official source:

```bash
cd ~ 
git clone git@github.com:NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
```

Apart from this, you will need approporiate versions of torch-scatter, torch-sparse, torch-spline-conv, torch-geometric, depending on your torch+cuda combination. E.g. for torch-1.13.1 + cuda11.7 you'd need:  

```commandline
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu117.html --force-reinstall
```

Maybe you will need to resolve the conflict using `pip check` command.

## Dataset

From project root execute:
```bash
mkdir data
cd data
wget https://www.dropbox.com/s/or9tfmunvndibv0/data.zip
unzip data.zip
```

For custom data processing check out https://github.com/nihalsid/CADTextures

## Output Directories

Create a symlink `runs` in project root from a directory `OUTPUTDIR` where outputs would be stored  
```bash
ln -s OUTPUTDIR runs
```

## Running Experiments

Configuration provided with hydra config file `config/stylegan2.yaml`. Example training:

```bash
python trainer/train_stylegan_real_feature.py 
```

## Author's Checkpoints

Available [here](https://www.dropbox.com/scl/fi/cz9arygdbz05gucapldd1/texturify_checkpoints.zip?rlkey=n19t8x0zq13i7hmodfnjkgrst&dl=0).

## Our Checkpoints

Available [here](https://drive.google.com/file/d/13K_KxlDT1esfVII78_tpgUAuS59rjeWs/view?usp=drive_link).

To test our checkpoints, create a folder named `checkpoints` under the project root directory.
Unzip the aforementioned checkpoints file into the `checkpoints` folder. 
Ensure that the folder structure matches the following diagram.

```
Texturify
│ 
├── checkpoints
│   ├── baseline
│   ├── class feature network
│   ├── D_gen
│   ├── F_latent
│   └── F_sum
└── evaluation
    ├── compute_metrics_combined_discriminator.py
    ├── compute_metrics_combined_latent.py
    ├── compute_metrics_combined_sum.py
    └── compute_metrics_combined_vanila.py
```

Use the following command under the root folder to evaluate our checkpoints.
All the results in this project were generated using 2 NVIDIA RTX 3090 GPUs.

```
python evaluation/compute_metrics_combined_sum.py
```

## Configuration

<details>
<summary>Configuration can be overriden with command line flags.</summary>

| Key | Description | Default |
| ----|-------------|---------|
|`dataset_path`| Directory with processed data||
|`mesh_path`| Directory with processed mesh (highest res)||
|`pairmeta_path`| Directory with metadata for image-shape pairs (photoshape specific)||
|`image_path`| real images ||
|`mask_path`| real image segmentation masks ||
|`uv_path`| processed uv data (for uv baseline) ||
|`silhoutte_path`| texture atlas silhoutte data (for uv baseline) ||
|`experiment`| Experiment name used for logs |`fast_dev`|
|`wandb_main`| If false, results logged to "<project>-dev" wandb project (for dev logs)|`False`|
|`num_mapping_layers`| Number of layers in the mapping network |2|
|`lr_g`| Generator learning rate | 0.002|
|`lr_d`| Discriminator learning rate |0.00235|
|`lr_e`| Encoder learning rate |0.0001|
|`lambda_gp`| Gradient penalty weight | 0.0256 |
|`lambda_plp`| Path length penalty weight |2|
|`lazy_gradient_penalty_interval`| Gradient penalty regularizer interval |16|
|`lazy_path_penalty_after`| Iteration after which path lenght penalty is active |0|
|`lazy_path_penalty_interval`| Path length penalty regularizer interval |4|
|`latent_dim`| Latent dim of starting noise and mapping network output |512|
|`image_size`| Size of generated images |64|
|`num_eval_images`| Number of images on which FID is computed |8096|
|`num_vis_images`| Number of image visualized |1024|
|`batch_size`| Mini batch size |16|
|`num_workers`| Number of dataloader workers|8|
|`seed`| RNG seed |null|
|`save_epoch`| Epoch interval for checkpoint saves |1|
|`sanity_steps`| Validation sanity runs before training start |1|
|`max_epoch`| Maximum training epochs |250|
|`val_check_interval`| Epoch interval for evaluating metrics and saving generated samples |1|
|`resume`| Resume checkpoint |`null`|

</details>


References
==========
Official stylegan2-ada code and paper.

```
@article{Karras2019stylegan2,
    title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
    author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
    journal = {CoRR},
    volume  = {abs/1912.04958},
    year    = {2019},
}
```

Original Texturify.

```
@inproceedings{siddiqui2022texturify,
  author    = {Yawar Siddiqui and Justus Thies and Fangchang Ma and Qi Shan and Matthias Nie{\ss}ner and Angela Dai},
  editor    = {Shai Avidan and Gabriel J. Brostow and Moustapha Ciss{\'{e}} and Giovanni Maria Farinella and Tal Hassner},
  title     = {Texturify: Generating Textures on 3D Shape Surfaces},
  booktitle = {Computer Vision - {ECCV} 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part {III}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13663},
  pages     = {72--88},
  publisher = {Springer},
  year      = {2022},
  url       = {https://doi.org/10.1007/978-3-031-20062-5\_5},
  doi       = {10.1007/978-3-031-20062-5\_5},
  timestamp = {Tue, 15 Nov 2022 15:21:36 +0100},
  biburl    = {https://dblp.org/rec/conf/eccv/SiddiquiTMSND22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

License
=====================

Copyright © 2021 nihalsid

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

