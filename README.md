# Sign Language Classification

In `sign-image-classifier` you can find a sign classifier based on a CNN

In `video-classifier` you can find 2 video classification models

1. Extract features from each frame with a ConvNet, passing the sequence to an RNN, in a separate network
2. Use a time-dstirbuted ConvNet, passing the features to an RNN 

# Requirements
The code was run using Cudnn v6, Tensorflow 1.4.1 and Keras 2.1.2
