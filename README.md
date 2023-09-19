## StyleGAN2-ADA BMC implementation

**Stabilizing GANs’ Training with Brownian Motion Controller**<br>
Tianjiao Luo, Ziyu Zhu, Jianfei Chen, Jun Zhu<br>
https://arxiv.org/pdf/2306.10468.pdf<br>

Abstract: *The training process of generative adversarial networks (GANs) is unstable and does not converge globally. In this paper, we examine the stability of GANs from the perspective of control theory and propose a universal higher-order noisebased controller called Brownian Motion Controller (BMC). Starting with the prototypical case
of Dirac-GANs, we design a BMC to retrieve
precisely the same but reachable optimal equilibrium. We theoretically prove that the training
process of DiracGANs-BMC is globally exponential stable and derive bounds on the rate of convergence. Then we extend our BMC to normal
GANs and provide implementation instructions
on GANs-BMC. Our experiments show that our
GANs-BMC effectively stabilizes GANs’ training
under StyleGANv2-ada frameworks with a faster
rate of convergence, a smaller range of oscillation,
and better performance in terms of FID score.*

### This implementation includes Dirac-GAN-BMC and styleGAN2-ada-BMC with controller on generator
### This repo is modified from styleGAN2-ADA-pytorch implementation 


## Note
Contact luotj21@mails.tsinghua.edu.cn if you have any questions

## Instructions on BMC
- Dirac-GAN folder includes code to reproduce figure 1 in our paper
- The Brownian Motion variable B(t) follows a Gaussian distribution
- The mean and variance of B(t) are hyperparameters to be changed 
- The controller can be added to either generator or discriminator or both depending on the nature of GAN's skeleton
- For WGAN series we add BMC on discriminator and for styleGAN we add BMC on geneator 

## Getting started

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames or URLs:

```.bash
# Generate curated MetFaces images without truncation (Fig.10 left)
python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

# Generate uncurated MetFaces images with truncation (Fig.12 upper left)
python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

# Generate class conditional CIFAR-10 images (Fig.17 left, Car)
python generate.py --outdir=out --seeds=0-35 --class=1 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

# Style mixing example
python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.

**Docker**: You can run the above curated image example using Docker as follows:

```.bash
docker build --tag sg2ada:latest .
./docker_run.sh python3 generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
```

Note: The Docker image requires NVIDIA driver release `r455.23` or later.

**Legacy networks**: The above commands can load most of the network pickles created using the previous TensorFlow versions of StyleGAN2 and StyleGAN2-ADA. However, for future compatibility, we recommend converting such legacy pickles into the new format used by the PyTorch version:

```.bash
python legacy.py \
    --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \
    --dest=stylegan2-cat-config-f.pkl
```

## Projecting images to latent space

To find the matching latent vector for a given image file, run:

```.bash
python projector.py --outdir=out --target=~/mytargetimg.png \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

For optimal results, the target image should be cropped and aligned similar to the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset). The above command saves the projection target `out/target.png`, result `out/proj.png`, latent vector `out/projected_w.npz`, and progression video `out/proj.mp4`. You can render the resulting latent vector by specifying `--projected_w` for `generate.py`:

```.bash
python generate.py --outdir=out --projected_w=out/projected_w.npz \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

## Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
```

The above code requires `torch_utils` and `dnnlib` to be accessible via `PYTHONPATH`. It does not need source code for the networks themselves &mdash; their class definitions are loaded from the pickle via `torch_utils.persistence`.

The pickle contains three networks. `'G'` and `'D'` are instantaneous snapshots taken during training, and `'G_ema'` represents a moving average of the generator weights over several training steps. The networks are regular instances of `torch.nn.Module`, with all of their parameters and buffers placed on the CPU at import and gradient computation disabled by default.

The generator consists of two submodules, `G.mapping` and `G.synthesis`, that can be executed separately. They also support various additional options:

```.python
w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
img = G.synthesis(w, noise_mode='const', force_fp32=True)
```

Please refer to [`generate.py`](./generate.py), [`style_mixing.py`](./style_mixing.py), and [`projector.py`](./projector.py) for further examples.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels.

Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

Legacy TFRecords datasets are not supported &mdash; see below for instructions on how to convert them.

**FFHQ**:

Step 1: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as TFRecords.

Step 2: Extract images from TFRecords using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
python ../stylegan2-ada/dataset_tool.py unpack \
    --tfrecord_dir=~/ffhq-dataset/tfrecords/ffhq --output_dir=/tmp/ffhq-unpacked
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq.zip

# Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq256x256.zip \
    --width=256 --height=256
```

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and create ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/metfaces/images --dest=~/datasets/metfaces.zip
```

**AFHQ**: Download the [AFHQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and create ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/afhq/train/cat --dest=~/datasets/afhqcat.zip
python dataset_tool.py --source=~/downloads/afhq/train/dog --dest=~/datasets/afhqdog.zip
python dataset_tool.py --source=~/downloads/afhq/train/wild --dest=~/datasets/afhqwild.zip
```

**CIFAR-10**: Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

**LSUN**: Download the desired categories from the [LSUN project page](https://www.yf.io/p/lsun/) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/lsun/raw/cat_lmdb --dest=~/datasets/lsuncat200k.zip \
    --transform=center-crop --width=256 --height=256 --max_images=200000

python dataset_tool.py --source=~/downloads/lsun/raw/car_lmdb --dest=~/datasets/lsuncar200k.zip \
    --transform=center-crop-wide --width=512 --height=384 --max_images=200000
```

**BreCaHAD**:

Step 1: Download the [BreCaHAD dataset](https://figshare.com/articles/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186).

Step 2: Extract 512x512 resolution crops using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
python dataset_tool.py extract_brecahad_crops --cropsize=512 \
    --output_dir=/tmp/brecahad-crops --brecahad_dir=~/downloads/brecahad/images
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
python dataset_tool.py --source=/tmp/brecahad-crops --dest=~/datasets/brecahad.zip
```

## Training new networks

In its most basic form, training new networks boils down to:

```.bash
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 --dry-run
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1
```

The first command is optional; it validates the arguments, prints out the training configuration, and exits. The second command kicks off the actual training.

In this example, the results are saved to a newly created directory `~/training-runs/<ID>-mydataset-auto1`, controlled by `--outdir`. The training exports network pickles (`network-snapshot-<INT>.pkl`) and example images (`fakes<INT>.png`) at regular intervals (controlled by `--snap`). For each pickle, it also evaluates FID (controlled by `--metrics`) and logs the resulting scores in `metric-fid50k_full.jsonl` (as well as TFEvents if TensorBoard is installed).

The name of the output directory reflects the training configuration. For example, `00000-mydataset-auto1` indicates that the *base configuration* was `auto1`, meaning that the hyperparameters were selected automatically for training on one GPU. The base configuration is controlled by `--cfg`:

| Base config           | Description
| :-------------------- | :----------
| `auto`&nbsp;(default) | Automatically select reasonable defaults based on resolution and GPU count. Serves as a good starting point for new datasets but does not necessarily lead to optimal results.
| `stylegan2`           | Reproduce results for StyleGAN2 config F at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `paper256`            | Reproduce results for FFHQ and LSUN Cat at 256x256 using 1, 2, 4, or 8 GPUs.
| `paper512`            | Reproduce results for BreCaHAD and AFHQ at 512x512 using 1, 2, 4, or 8 GPUs.
| `paper1024`           | Reproduce results for MetFaces at 1024x1024 using 1, 2, 4, or 8 GPUs.
| `cifar`               | Reproduce results for CIFAR-10 (tuned configuration) using 1 or 2 GPUs.

The training configuration can be further customized with additional command line options:

* `--aug=noaug` disables ADA.
* `--cond=1` enables class-conditional training (requires a dataset with labels).
* `--mirror=1` amplifies the dataset with x-flips. Often beneficial, even with ADA.
* `--resume=ffhq1024 --snap=10` performs transfer learning from FFHQ trained at 1024x1024.
* `--resume=~/training-runs/<NAME>/network-snapshot-<INT>.pkl` resumes a previous training run.
* `--gamma=10` overrides R1 gamma. We recommend trying a couple of different values for each new dataset.
* `--aug=ada --target=0.7` adjusts ADA target value (default: 0.6).
* `--augpipe=blit` enables pixel blitting but disables all other augmentations.
* `--augpipe=bgcfnc` enables all available augmentations (blit, geom, color, filter, noise, cutout).

Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list.

