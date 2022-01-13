# Brain-Tumor-Classification

![Brain Tumor Classification](https://user-images.githubusercontent.com/81551950/149251288-6f45a8be-1027-4e6c-b910-b29dfc3aacbc.png)

## Abstract

The brain tumor is one of the most destructive diseases leading to short life expectancy in the highest grade. Of all primary Central Nervous System (CNS) tumors, brain tumors account for 85 to 90 percent. The misdiagnosis of brain tumors will result in wrong medical treatment, which in turn results in reducing the chance of survival. In order to increase the life expectancy of patients, adequate care, preparation, and reliable diagnostics should be introduced. Magnetic Resonance Imaging (MRI) is the best way to identify brain tumors. Through the scans, a huge amount of image data is produced. There are several anomalies in the brain tumor size and location (s). This makes it very difficult to completely comprehend the nature of the tumor. For MRI analysis, a trained neurosurgeon is also needed. The lack of knowledge about tumors also makes it very difficult and time-consuming to produce MRI studies. An automated system can solve this problem. Application of automated classification techniques using Machine Learning (ML) and Artificial Intelligence (AI) has consistently shown higher accuracy than manual classification. It would be beneficial to propose a method for classification using Deep Learning Algorithms. In the proposed framework, we conduct three studies using three architectures of neural networks, convolutional neural networks (Alex Net, ResNet 50, and VGGNet), to classify brain tumors such as meningioma, glioma, and pituitary and explored transfer learning techniques.


## Data Description and EDA

The dataset contains 2881 train and 402 test MRI scanned grayscale images that further falls into four categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor. Glioma tumor occurs in the brain and spinal cord, whereas Pituitary tumor occurs in the pituitary gland. Meningioma tumor arises from the membranes surrounding the brain and spinal cord. After analyzing the provided Data, the partition for the training and testing data set is presented visually below.



## Data Preprocessing

Before building a model on the provided data, we preprocessed it by applying the following steps.
1. Reading the images from all four folders and storing them in the array, separately as training and testing data
2. Reshaping the images to 150x150 pixels to maintain the same size
3. To convert the range of grayscale images from 0-255 to 0-1, each image was divided by 255
4. One-Hot Encoding of Test Images so that the output could be evaluated



## Proposed Approach
Deep Learning Model is implemented to analyze and classify the images. Once trained, this model could be generalized to classify other images into various brain tumor types or no brain tumor types.

## Neural Network Hyperparameters
Neural Network Hyperparameters11 can be categorised in 3 main categories
1. Network Architecture
* No of Hidden Layers (Network Depth)
* No of Neurons in each Layer (Layer Width)
* Activation Type

2. Learning and Optimisation
* Learning Rate and Decay Schedule
* Batch Size
* Optimisation Algorithms
* Epochs and Early-Stopping Criteria

3. Regularisation Techniques to Avoid Overfitting
* L2 Regularisation
* Dropout Layers
* Data Augmentation



### ANN Model
Various authors adopted artificial neural networks (ANNs) to optimize multipurpose parameters. In most cases ANN
allows to predict the properties of the dataset or model. In our study we used different ANN model and the comparison
of them is given in Table 1.

### CNN Model
In our study, we proposed a simple CNN model, we extracted the augmented MRI image data of 150 × 150 input size
having Greyscale with a batch size of 32 through our CNN model. Initially, we added a single 16 filters convolutional
layer having a filter size of 3 × 3. The reason for placing a small number of filters as 16 is to detect edges, corners,
and lines. Table 2 displays the comparison of various various CNN architecture that we have implemented.





## Transfer Learning
In deep learning, sometimes we use a transfer learning approach in which instead of making a scratched CNN model for the image classification problem, a pre-trained CNN model that is already modeled on a huge benchmark dataset, like ImageNet, is reused. Sinno Pan and Qiang Yang have introduced a framework for a better understanding of Transfer Learning. Instead of starting the learning process from scratch, the transfer learning leverages previous learning. We started with Transfer Learning Model AlexNet that was proposed by Alex Krizhevsky in 2012. However, the results of the model were really low for our dataset so we explored and applied more transfer learning models.

### 1. VGG16
We used a pre-trained VGG-16 convolutional neural network model which is fine tuned by freezing some of the layers to avoid overfitting because our dataset is very small. VGG16 is a CNN model of 16 Convolutional layers proposed in 2014 by Karen Simonyan and Andrew Zisserman. The network image input shape is 150 × 150 × 3. It includes 16 Convolution layers with a fixed 3 × 3 filter size and 5 Max pooling layers of 2 × 2 size throughout the network. And at the top the 2 fully connected layers with a softmax output layer. VGG-16 Model is a large network, with approximately 138 million parameters. It’s stacking many convolutional layers to build deep neural networks that improve the ability to learn hidden features.

### 2. Densenet201
It is a convolutional neural network that is 201 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.


### 3. ResNet152V2
It is a convolutional neural network that is multiple layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

### 4. Squeeznet
SqueezeNet is the name of a deep neural network for computer vision that was released in 2016. SqueezeNet was developed by researchers at DeepScale, University of California, Berkeley, and Stanford University. In designing SqueezeNet, the authors' goal was to create a smaller neural network with fewer parameters that can more easily fit into computer memory and can more easily be transmitted over a computer network.

### 5. InceptionV3
Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).

### 6. InceptionResnetV2
It is a convolutional neural network that is trained on more than a million images from the ImageNet database. The network is 164 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 299-by-299




## Conclusions
1. Multiple Transfer Learning Models have been implemented to compare the outcome
2. Among all models we applied, 6 models had good accuracy
3. Maximum validation accuracy is 79% for DenseNet201
