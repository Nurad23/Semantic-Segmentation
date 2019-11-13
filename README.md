# Semantic-Segmentation
Ship Detection and Segmentation


This project aims to correctly segmentate ships from satellite images using Deep Learning. The dataset has roughly around 100,000 images.
The model used is a modified version of U-net and trained on Amazon EC2 using Cuda. The detailed information about the problem, model and report can be 
found at Report.pdf document.

# Data Preprocessing 

The ratio of samples having ship and not having any ship was 4:1 initially, it was reduced to the ratio of 3:2. The final dataset consists of 109,000 training images with 60% of images having no ships and rest 40% having one or multiple ships.


# Data Augmentation

For Data Augmentation, we used the Keras Image generator function and a custom data augmentation function. It consists of performing different transformations on existing images in order to obtain images that make the machine learning model more robust. First, we normalize the images by dividing each pixel by 255 (maximum range of the pixel value) and additionally, we flip the image, increase the brightness and resize it to reduce complexity and computational power.


# Model

Each layer in the contraction path consists of 2-3x3 convolution layers followed by a ReLu layer and a 2x2 max pooling operation afterwards. There is a bottleneck layer between the contraction and expanding path which consists of 2 convolutions without max pooling. The expanding path starts with transposed convolution layer where the image size is doubled for reconstruction and is followed by concatenation (skip connection) with the corresponding convolutional layer in the contraction path.

# Training Process

To get the best working model in our use case we tried different U-Net architectures and we tuned the hyperparameters accordingly. The model is trained using Amazon p2.8xlarge EC2 instance using Tensorflow with CUDA.
