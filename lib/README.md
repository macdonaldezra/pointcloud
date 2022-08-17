# PyTorch Wrapped Functions

Please Note:
This has been retrieved from [point-transformer](https://github.com/POSTECH-CVLab/point-transformer) with the exception that the library has been minorly modified to be compatible with PyTorch v1.12

To install the CUDA functions in this directory, run the following (assuming you have a NVIDIA GPU with architecture 6.1):

```bash
TORCH_CUDA_ARCH_LIST="6.1" MAX_JOBS=4 FORCE_CUDA=1 python3 setup.py install
```

Note that the value for TORCH_CUDA_ARCH_LIST must match the architecture for the GPU that the functions are compiled on whicch can be found [here](https://developer.nvidia.com/cuda-gpus#compute)
