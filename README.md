# PointCloud DL

A repository for training and building PointCloud deep learning models with

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
