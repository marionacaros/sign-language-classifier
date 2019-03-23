# Sign video classification 

Two video classification methods are presented:

1. Extract features from each frame with a ConvNet, passing the sequence to an RNN, in a separate network
1. Use a time-dstirbuted ConvNet, passing the features to an RNN (this is the `lrcn` network in the code).


## Requirements

The code was run using Cudnn v6, Tensorflow 1.4.1 and Keras 2.1.2

To ensure you're up to date, run: `pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files. If `ffmpeg` isn't in your system path (ie. `which ffmpeg` doesn't return its path, or you're on an OS other than *nix), you'll need to update the path to `ffmpeg` in `data/2_extract_files.py`.

## Getting the data

A small dataset containing sign language videos is already downloaded in `data/train`, `data/test`, `data/true-test`. 
The function that has been used to split the data is `move_files.py` and to extract the frames from the video: `extract_files.py`. 
More data can be downloaded to be used in the model by using these functions.

## Extracting features

Before you can run the `lstm`, you need to extract features from the images with the CNN. This is done by running `extract_features.py`. 

## Training models

The models are run from `train.py`. There are configuration options you can set in that file to choose which model you want to run.

The models are all defined in `models.py`. Reference that file to see which models you are able to run in `train.py`.

Training logs are saved to CSV and also to TensorBoard files. To see progress while training, run `tensorboard --logdir=data/logs` from the project root folder.

## Eval

There is an evaluation function implemented to evaluate the accuracy of the selected model.

## Demo

There is a demo implemented to see the predicted results of the model. 


## Citation

Code is based on: https://medium.com/@harvitronix/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5 