# PointCloud DL

A repository for training and building PointCloud deep learning models with PyTorch. This repository currently contains data loaders for the [Sensat Urban](https://paperswithcode.com/dataset/sensaturban) and [ShapeNet](https://shapenet.org/) datasets as well as implementations of the [Point-Transformer](https://github.com/POSTECH-CVLab/point-transformer) and [PointNet](https://paperswithcode.com/paper/pointnet-deep-learning-on-point-sets-for-3d) models.

## Install

This project uses [Pyenv](https://github.com/pyenv/pyenv#installation) for managing the Python version and environment. To create the virtual environment and install dependencies run the following:

```bash
pyenv virtualenv 3.9.1 segmentation-env
pyenv activate segmentation-env
pip install requirements.txt
```

### Configure Pre-Commit Hooks

To better ensure that only properly formatted Python code is pushed to this repository, we use pre-commit hooks. To configure [pre-commit](https://pre-commit.com/) hooks run the following command:

```bash
pre-commit install
# Run against all files that are currently committed in the project
pre-commit run --all-files
```

## Data Sets

So far there are data loaders for ShapeNet and SenSat Urban datasets.

### SensatUrban

Learn more about the SensatUrban dataset [here](https://paperswithcode.com/dataset/sensaturban). It is perhaps worth noting here that the test dataset that SensatUrban has provided does not include labels.


#### Pre-Processing Data

Once you have installed the requirements to pre-process the [Sensat Urban](https://paperswithcode.com/dataset/sensaturban) dataset to create the sampled pointcloud points, you can run the following commands:

```bash
# Compile the functions that are required for performance-enhanced grid-subsampling functions
# found in the cpp_wrappers directory.
./cpp_wrappers/compile_wrappers.sh
# Note that all of the parameters are optional
# Also note that the SensatUrban dataset is expected to have train/ and test/ subdirectories
python -m pointcloud.processors.sensat.prepare_inputs --data-path <path_to_data_directory_root> \
    --grid-size 0.3 \
    --leaf-size 50
```

**Please Note**: Choosing a `--grid-size` of 0.1 resulted in outputted pointcloud files that were larger than the original input files, hence you should probably use a grid size of about 2 or greater.

### Models

So far there is a PointNet and PointTransformer found in this repository, with more to come in all likelihood.

### Point-Transformer

The Point-Transformer model in this repository has been retrieved from the official [Point-Transformer](https://github.com/POSTECH-CVLab/point-transformer) repository that accompanied the [paper](https://paperswithcode.com/paper/point-transformer-1). This repository requires special PyTorch functions to be compiled before using this model. To compile these please consult the instructions found in `lib/`.
