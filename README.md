MODEL:
Convolutional Neural Networks (CNNs) are a type of artificial neural network that are specifically designed to process and classify images. They have gained popularity in recent years due to their impressive performance in a variety of computer vision tasks, such as object recognition, image segmentation, and facial recognition. In this project, we developed a CNN using Python programming language to classify images from the CIFAR-10 dataset.
The CNN architecture is based on the principles of convolution and pooling. Convolution is a mathematical operation that involves sliding a small window, called a filter or kernel, over the image and computing the dot product between the filter and the corresponding pixels in the image. This produces a feature map that highlights specific patterns in the image. Pooling is another operation that involves downsampling the feature maps to reduce their size and extract the most important information. The model consists of several layers, including convolutional layers, pooling layers, and fully connected layers. The convolutional layers perform the convolution operation on the input image, while the pooling layers downsample the feature maps. The fully connected layers combine the extracted features and produce the final output.

DATASET:
The dataset used in this project is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes. The dataset was preprocessed by normalizing the pixel values to be between 0 and 1 and performing one-hot encoding on the labels. The model was trained using a batch size of 4 and a learning rate of 0.001 for 50 epochs. The batch files used has been downloaded from cifar-10 page, there are 5 batches for the training, 1 to test and the last one with the metadata of the dataset. The images have been transformed in Pil images, they are resized and normalized, then converted to tensors as input for the CNN.

RESULTS:
The final accuracy of the model on the test set was 50%. This means that the model was able to correctly classify 50% of the images in the test set. 

CONCLUSION:
The architecture of the CNN used in this project was relatively simple, yet it was able to achieve good results on a small dataset. Further improvements could be made by using more complex architectures and augmenting the dataset with additional images.
