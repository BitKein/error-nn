#!/bin/bash

# Job name
#SBATCH --job-name=test-amartin
#sbatch --job-name=test

# Define the files which will contain the Standard and Error output
#SBATCH --output=outputs/M_%A.out
#SBATCH --error=outputs/M_%A.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
# #SBATCH --cpus-per-task=20
# #SBATCH --threads-per-core=2
# #SBATCH --ntasks-per-core=2

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=2G
#sbatch --mem-per-cpu=2G

# Select one partition
#SBATCH --partition=EHP-PAR
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1
#sbatch --partition=EHP-PAR
#sbatch --gpus-per-task=1
#sbatch --gpus=1

# If you are using arrays, specify the number of tasks in the array
#SBATCH --array=1-1

#cmd
./bin/nn --test  --dataset datasets/CC_test.csv --model model.m --verbose
