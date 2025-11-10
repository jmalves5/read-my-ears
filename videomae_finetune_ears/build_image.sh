#!/usr/bin/env bash

#SBATCH --job-name=build_image
#SBATCH --partition=prioritized
#SBATCH --nodelist=a256-t4-04
# #SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks=1

srun singularity -v  build --fakeroot pytorch.sif pytorch.def
