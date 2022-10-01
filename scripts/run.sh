#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --output=%N-%j.out

DATA_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
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
module load singularity/3.8
module load cuda/11.2.2 cudnn/8.2.0

mkdir -p /scratch/$USER/singularity/{cache,tmp}
export SINGULARITY_CACHEDIR="/scratch/$USER/singularity/cache"
export SINGULARITY_TMPDIR="/scratch/$USER/singularity/tmp"

# Set path for config file and start training the model
export SINGULARITYENV_MODEL_CONFIG="/data/config.yaml"
singularity run --nv --pwd /code \
  --bind $DATA_PATH:/data \
  --bind $OUTPUT_PATH:/output \
  train.image
