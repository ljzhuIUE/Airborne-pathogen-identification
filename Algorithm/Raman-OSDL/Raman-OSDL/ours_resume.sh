#!/bin/bash
#SBATCH -p HighPerformanceNode    
#SBATCH -N 1
#SBATCH -n 128
python ours.py --resume checkpoints/cifar/ResNet18/7zhe/last_model.pth --evaluate --test_path test_path_data/new_data/train_test --other_path other_path_data/new_data/other
