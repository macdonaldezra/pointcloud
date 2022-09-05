# PyTorch Wrapped Functions

Please Note:
This has been retrieved from [point-transformer](https://github.com/POSTECH-CVLab/point-transformer) with the exception that the library has been minorly modified to be compatible with PyTorch v1.12

To install the CUDA functions in this directory, run the following (assuming you have a NVIDIA GPU with architecture 6.1):

```bash
# Please ensure you have installed the required Python dependencies before
# by following the instructions found in the README in the root directory.
cd lib/pointops # must be in the directory containing setup.py
TORCH_CUDA_ARCH_LIST="6.1" MAX_JOBS=4 FORCE_CUDA=1 python setup.py install
```

Note that the value for TORCH_CUDA_ARCH_LIST must match the architecture for the GPU that the functions are compiled on whicch can be found [here](https://developer.nvidia.com/cuda-gpus#compute).
