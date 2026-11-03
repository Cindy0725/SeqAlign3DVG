# SeqAlign3DVG

Implementation of **SeqAlign3DVG: A Sequence-Aligned Benchmark and Voxel Reasoning Framework for 3D Visual Grounding**, ACM Multimedia, 2026.

[Paper](https://arxiv.org/abs/2608.30451)

SeqAlign3DVG introduces an observation-aligned benchmark for single-view and observation-sequence 3D visual grounding. The model combines Relevance-Ordered Voxel Memory (ROVM) with Progressive Language-Voxel Fusion (PLVF) to localize objects from RGB observations, camera parameters, and referring expressions.

This repository is an **MMDetection3D v1.2.0 project**. Place it at `mmdetection3d/projects/SeqAlign3DVG`.

## Installation

Use a dedicated Linux environment with an NVIDIA GPU. The versions below follow the development environment.

| Component | Version |
| --- | --- |
| Python | 3.8 |
| PyTorch / torchvision | 2.0.0+cu117 / 0.15.0+cu117 |
| CUDA toolkit | 11.7 |
| MMEngine / MMCV / MMDetection | 0.10.4 / 2.0.1 / 3.0.0 |
| MMDetection3D | v1.2.0 |
| xFormers | 0.0.18 |

### 1. Arrange the repositories

Download this repository into a directory named `SeqAlign3DVG`. From its parent directory:

```bash
git clone --branch v1.2.0 --depth 1 https://github.com/open-mmlab/mmdetection3d.git
cp -a SeqAlign3DVG mmdetection3d/projects/
cd mmdetection3d
```

Run subsequent commands from the **MMDetection3D root**, unless stated otherwise.

### 2. Create the environment

```bash
conda create -n seqalign3dvg python=3.8 -y
conda activate seqalign3dvg
python -m pip install pip==25.0.1 setuptools==60.2.0 wheel==0.45.1

python -m pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 \
  --index-url https://download.pytorch.org/whl/cu117

conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.5.3 \
  scikit-image==0.21.0 plyfile==1.0.3 \
  pyre-extensions==0.0.23 typeguard==2.13.3
python -m pip install openmim==0.3.9
mim install mmengine==0.10.4
mim install mmcv==2.0.1
mim install mmdet==3.0.0
python -m pip install -v -e . --no-build-isolation
python -m pip install xformers==0.0.18 --no-deps
```

### 3. Enable DINOv2

Use the historical DINOv2 source below. Its installer requires Python >=3.9, so this Python 3.8 setup imports the source through `PYTHONPATH`.

```bash
SEQALIGN_PROJECT="$PWD/projects/SeqAlign3DVG"
mkdir -p "$SEQALIGN_PROJECT/third_party"
git clone https://github.com/facebookresearch/dinov2.git \
  "$SEQALIGN_PROJECT/third_party/dinov2_py38"
git -C "$SEQALIGN_PROJECT/third_party/dinov2_py38" checkout \
  c3c2683a13cde94d4d99f523cf4170384b00c34c
export PYTHONPATH="$SEQALIGN_PROJECT/third_party/dinov2_py38:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Repeat the CUDA and `PYTHONPATH` exports when opening a new terminal.

### 4. Install CLIP

```bash
python -m pip install ftfy regex tqdm
git clone https://github.com/openai/CLIP.git clip_openai
cd clip_openai
python -m pip install -e . --no-deps
cd ..
```

The model uses DINOv2 ViT-S/14 and CLIP ViT-B/32. Their pretrained weights are downloaded automatically on first use if absent from the local cache.

## Data

We propose the **SeqAlign3DVG** dataset for single-view and observation-sequence 3D visual grounding. The complete dataset and download instructions will be released separately.

The current configurations expect the following layout:

```text
data/SeqAlign3DVG/
  scannet_3dvg_trainval_multiview_train_dc.pkl
  scannet_3dvg_trainval_multiview_val_dc.pkl
  scannet_3dvg_trainval_singleview_train.pkl
  scannet_3dvg_trainval_singleview_val.pkl
  posed_images/<scene>/
    <frame>.jpg
    <frame>.txt
    intrinsic.txt
```

The default `data_root = 'data/SeqAlign3DVG/'` is relative to the **MMDetection3D root**. If you store the dataset elsewhere, update `data_root` in `configs/seqalign3dvg.py` (multi-view) and `configs/seqalign3dvg_singleview.py` (single-view) to the actual dataset directory, keeping the trailing slash. Absolute paths are also supported.

RGB frames, camera poses/intrinsics, and referring-expression annotations are required; depth maps are not used. Ground-truth boxes are included in the annotations for training and evaluation.

## Evaluation

Download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1IkNQF95xHMnPLnLnfcfPRh0scoDpkZw8?usp=sharing). Save the corresponding weights as `checkpoints/seqalign3dvg_multiview.pth` and `checkpoints/seqalign3dvg_singleview.pth` under the MMDetection3D root.

After setting `data_root`, run:

```bash
# Multi-view test
python tools/test.py \
  projects/SeqAlign3DVG/configs/seqalign3dvg.py \
  checkpoints/seqalign3dvg_multiview.pth

# Single-view test
python tools/test.py \
  projects/SeqAlign3DVG/configs/seqalign3dvg_singleview.py \
  checkpoints/seqalign3dvg_singleview.pth
```

Evaluation reports Acc@0.25 and Acc@0.50 on the corresponding validation split.

## Citation

```bibtex
@inproceedings{zhang2026seqalign3dvg,
  title={SeqAlign3DVG: A Sequence-Aligned Benchmark and Voxel Reasoning Framework for 3D Visual Grounding},
  author={Zhang, Yi and Wang, Yi and Wu, Yueting and Yang, Kaiyue and Su, Yuejiao and Chau, Lap-Pui},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  doi={10.1145/3767308.3835644}
}
```

## Acknowledgments

This project builds on [MMDetection3D/ImVoxelNet](https://github.com/open-mmlab/mmdetection3d), [DINOv2](https://github.com/facebookresearch/dinov2), [CLIP](https://github.com/openai/CLIP), and [Atlas](https://github.com/magicleap/Atlas). We thank their authors for making the code available.
