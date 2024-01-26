# Airborne-pathogen-identification

Code repository for our paper ‘Open-Set Deep Learning-enabled Single Cell Raman for Rapid Identification of Real-World Airborne Pathogens’

## Abstract
Environments, particularly those with pathogenic bioaerosols, are critical in airborne transmission and epidemics outbreak. However, rapidly and accurately identifying pathogens directly in complex environments remains a big challenge. Here, we report the development of an open-set attentional neural network deep learning algorithm (OSDL) to enable single-cell Raman to identify pathogens in real-word air environments containing diverse, unknown indigenous air bacteria that cannot be totally included in the training dataset. Raman datasets of bacteria in aerosol state were also established to improve precise identification. The obtained Raman-OSDL system achieves 93% accuracy for five target airborne pathogens, to our knowledge the highest accuracy of 84% for unseen air bacteria, and a significant 36% reduction in false positive rates compared to conventional closed-set algorithm-based methods. It also offers a high sensitivity down to 1/1000 pathogen abundance. When deployed to real air samples containing 4600 bacteria species annotated by metagenomics, our system can accurately identify either single or multiple pathogens within this consortia, and the entire identification process can be completed in just one hour. This single-cell tool holds great promise in rapidly surveilling pathogens in real air and other environments, aiding in prevention of infection transmission.

## Requirements
python 3.6.15,\
pytorch 1.4+, torchvision 0.7.0 +, 
numpy 1.19.5, 
scikit-learn 0.22.1, 
libmr 0.1.9, 
cython 0.29.33, 
matplotlib 3.3.2, 
opencv-contrib-python 4.7.0.68, 
pandas 1.1.5, 
scipy 1.5.4

## Data
The dataset download link is https://www.scidb.cn/s/BrUnYr. The data used to train and test a new model should be placed under the path ‘…/Algorithm/Raman-OSDL/Raman-OSDL/data/new_data’. When evaluating a pretrained model, the data should be placed under the path ‘…/Algorithm/Raman-OSDL/Raman-OSDL/test_path_data/new_data’ (the classes appeared in the training set) or ‘…/Algorithm/Raman-OSDL/Raman-OSDL/other_path_data/new_data’ (the classes not appeared in the training set). When testing the real-world air samples, all the data are placed under the path ‘…/Algorithm/Raman-OSDL/Raman-OSDL/other_path_data/new_data’

## Model
The pretrained model were stored under the path ‘…/Algorithm/Raman-OSDL/Raman-OSDL/checkpoints/models’, ‘0-9’ means different models from the 10-fold cross validation.

## Running
For training and testing a new model: python main_program.py. 
For evaluating a pretrained model: python main_program.py --resume checkpoints/models/ANN/4_fold/last_model.pth --evaluate --test_path test_path_data/new_data --other_path other_path_data/new_data (taking the example of using the four-fold model for evaluation). 

Some parameters:
--resume: PATH. Load pretrained model to continue training.
--train_class_num: the number of known classes for training.
--test_class_num: the number of total classes for testing.
--evaluate: evaluate the model without training. So you should use --resume to load pretrained model.
--weibull_tail: parameters for weibull distribution, default 20.
--weibull_alpha: parameters for weibull distribution, default 3.
--weibull_threshold: parameters for confidence threshold, default 0.98. (0.98 may be the best for Raman-OSDL datasets)
