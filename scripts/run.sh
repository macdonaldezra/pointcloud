#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --output=%N-%j.out

# set -eo pipefail

EPOCHS=2
DATA_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--data-directory)
      DATA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-directory)
      OUTPUT_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      shift # past argument
      ;;
  esac
done


# Find latetst Singularity module on Compute Canada by running 'module spider singularity'
# module purge
module load singularity/3.8
module load cuda/11.2.2 cudnn/8.2.0

mkdir -p /scratch/$USER/singularity/{cache,tmp}
export SINGULARITY_CACHEDIR="/scratch/$USER/singularity/cache"
export SINGULARITY_TMPDIR="/scratch/$USER/singularity/tmp"

# #
# # Pipe output to another file
# #
singularity exec --nv --pwd /code \
  --bind $DATA_PATH:/data --bind $OUTPUT_PATH:/output \
  train.image \
    python -m pointcloud.train.point_transformer \
      --epochs $EPOCHS \
      --data-directory /data \
      --output-directory /output
