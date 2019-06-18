# Deep Learning
## Image Classifier Project

## Project Overview
In this project, I trained an image classifier to recognize different species of flowers. I used tensorflow in order to train the model and classify the images. I ultimately achieved a 91.64% accuracy with the image classifier.

## Data
Unfortunately, the dataset will not be included in the repository as it is too big. The dataset is hosted at this website if you want to download it: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

## File Contents

    .
    ├── Image Classifier Project.ipynb               # Notebook workspace
    ├── Image Classifier Project.html                # HTML version of notebook workspace
    ├── predict.py                                   # Run this in console to classify image
    ├── prediction.py                 
    ├── train.py                                     # Run this in console to train classifier
    ├── train_model.py
    ├── checkpoint.pth                               # Saved Model
    ├── workspace_utils.py                      
    ├── LICENSE
    └── README.md
    
## Run Instructions (Command Line)

- Training the classifier
    - The training loss, validation loss, and validation accuracy are printed out as a network trains.
    - Basic Usage: python train.py data_directory
    - Optional Parameters:
        - Set direcotry to save checkpoints: python train.py data_dor --save_dir save_directory
        - Choose arcitecture: pytnon train.py data_dir --arch "vgg11"
        - Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 4096 --epochs 20
        - Use GPU for training: python train.py data_dir --gpu gpu
- Running the classifier
    - Will return the flower name and class probability.
    - Basic Usage: python predict.py /path/to/image checkpoint
    - Optional Parameters:
        - Return top K most likely classes: python predict.py input checkpoint --top_k 3
        - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
        - Use GPU for inference: python predict.py input checkpoint --gpu
