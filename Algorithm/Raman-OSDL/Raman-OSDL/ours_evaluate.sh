#!/bin/bash
#SBATCH -p HighPerformanceNode
#SBATCH -N 1
#SBATCH -n 2
python main_program.py --resume checkpoints/models/ANN/4折/last_model.pth --evaluate --test_path test_path_data/new_data/P7 --other_path other_path_data/new_data/other7
