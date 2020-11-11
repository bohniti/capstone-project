<div align="center"> 
<img src="https://images.unsplash.com/photo-1562054438-f789d60d03eb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60" width="700px;">
<h1>My first Deep Learning Project with PyTorch<h1>
<h2>Classification of DogBreeds<h2>
</div>


## Content

- Python Notebook as [iPython](https://github.com/bohniti/capstone-project/blob/master/dog-breed-classifier-final.ipynb)
- Report as [Medium Post](https://timo-bohnstedt.medium.com/my-first-deep-learning-project-with-pytorch-dogbreeds-5750fbc0f9da)

# Project Overview
The aim of my final project within Udacity Data Science Nano Degree was too learn how to apply Deep Learning in PyTorch. If a dog is detected in the image, it will provide an estimate of the dog’s breed. Thus, the idea is to use several training example such I can predict any dog breed from an arbitrary image.

# Problem Statement
Within this project I use Convolutional Neural Networks (CNNs) to classify images of dog breeds. The postulate for pattern recognition “compact domain” is not given such we have a high intra-class distance and a small inter-class distance. This makes it difficult to implement a classifier. For example some images look quite similar even there classes are different. This is why I will use transfer learning. The benefits of transfer learning are the it can speed up the time it takes to develop and train model. Therefore we use the weights which trained to recognize certain patterns such as edges. Then we have just to implement a step, where we fine tune the weights a little bit in order to get outstanding results.

# Metrics
As I said before the task I will work on is a classification problem. For obtaining a model I need to split the dataset into a training set, a validation set and a testing set. The model(s) I used were evaluated by using the accuracy as a metric:

<img src="https://miro.medium.com/max/518/1*IJZN8wPMLgm8zR1wuhwcuw.png" width="700px;">

# Data Exploration and visualization
## Datasets and Inputs
For this project, I used Stanford Dog Breed dataset which contains images of 120 dog breeds. The dataset contains more than 20000 images.The dataset has been built using images and annotation from ImageNet for the task of image classification. It was originally collected in order to challenge classifiers because the features of some breeds are almost identical.
In the following you can see a few samples from certain dog breeds and there class.

<img src="https://miro.medium.com/max/700/1*xKb0RdQgMStGgoAswBx4mA.png" width="700px;">

Notice that this image we need to match a labels to images. Usually if you are start to train you model you will vectorize the images and you change the class names by inter representations such as affenpinscher = 0.
If you had a look at the plot below you can see that the image size is not equal over the data set. We have to this while we are preprocessing our work.

<img src="hhttps://miro.medium.com/max/700/1*8lL_LVqxdfhNFXVcRgDSog.png" width="700px;">

# Methodology
## Data Preprocessing
A classic approach of preprocessing would contain feature extraction or a dimensionality reduction. Consider that we are in the field of deep learning that mean that we are not taking feature extraction in consideration because this is the advantage of deep learning. It decides for its self whether a feature is important or not.
PyTorch CNNs require a 4D tensor as input training example. A tensor can thought of a matrix. It size shall be the number of samples times the columns channels (pixels). The labels are a vector with a corresponding inter as described before.
There was a function built to create this array (tensor). First, it loads the image and resizes it to a square image. Next, the image is converted to an array, which is then resized to a 4D tensor.

# Implementation
To solve the image classification task, I was using a convolutional neuronal network (CNN). CNN’s are a particular case of feed-forward neural networks. On a very fundamental level, we I would say that a feed-forward neural network — as the most machine learning models — is a function.
I decided to use the VGG16 model architecture as a base.
In 2014, an implementation of the architecture won the ILSVR(Imagenet) competition.The complexity is rather small in comparison to other designs which are provided in the ranking. The main goal was to understand the principals behind deep learning and to get a feeling of what the state of the art (in image recognition) is. So, because the VGG16 is known as the best visualizable model architecture, we thought it would be the best choice to accomplish our goal in this way. As you can see the model consists of Pooling layers, fully connected layers and uses batch normalization.

<img src="https://miro.medium.com/max/656/1*yf8fX134uLhY9faIg5jALA.png" width="700px;">

# Refinement
CNN without transfer learning consumed lot of time and was not satisfactory in terms of performance. Thus to reduce training time without sacrificing accuracy, I trained a CNN using transfer learning. So I used different architectures in order to compare the results. For example with my VGG16 I had an accuracy of 70 percent where the transfered and more complicated ResNet architecture had 80 per cent accuracy.

# Results
## Model Evaluation and Validation
As I said before I used different used the accuracy as a metric in order to compare different models. Just for my own interesset I compared it to other architectures as well and derived the following table.

<img src="https://miro.medium.com/max/700/0*FWuN5X4X0Mv1DoTf.png" width="700px;">

# Justification
From my point of view the result is quite good. Because 70,2 percent of dog breeds were detected in the first 13 epochs with a very basic CNN architecutre (VGG16). But as we can see from the tabel above, that a more modern approach of classification model, will increase the training accuracy (after 2 epochs) reached over 60%. Augmentation would teach the model on images that are variations of the original dataset which allows the model to generalize even better.

# Reflection
The accuracy obtained was nearly 84% using CNN with transfer learning. Among the different bottleneck features, ResNet performed best. This result is quite impressive as compared to CNN from scratch with accuracy with 60%. As because ResNet model was built on the imagenet library and I am transferring that knowledge to my new problem. The imagenet libary is quite large (1.2 million images) and includes images of dogs which means the transferred learning is based on a similar dataset, at least partially.

# Improvement
There may be some possible points of improvement of the algorithm. One way is by applying data augmentaion to combat overfitting on training set. Secondly, by adding more breeds of dogs to the training and test data, so as to improve the breadth of identification and prediction. Only training has been done using 133 breeds and there are nearly 344 breeds according to the Fédération Cynologique Internationale. Thirdly, we can also try with a different network for feature extraction.

## Prerequisites

Create the conda environment:

``` 
    conda env create -f environment.yml
```



### Follow these steps:


1) Go to [Kaggle](https://www.kaggle.com/c/dog-breed-identification) and get the Dataset.

2) Save it within you directory in the input folder as describet in the structure above.

3) Just Run the Notebook.


## Workflow

When you click the button *Create Scheduler for next week*, an entry in the *Data/scheduler_df.csv* will be created for the upcoming week.
For that reason you cannot create multiple events for the upcoming week. If you wanna re-do the event adding, just go into the *scheduler_df.csv* 
and set the True to False.


## Run the Notebook

```
    cd/you_cloned_repo_location/Notebooks jupyter notebook
```

## Deployment

Just pull the repo, if you wanna change sth you can ask :)

## Authors

* **Timo Bohnstedt** - [GitHub Bohniti](https://github.com/bohniti)



## License

MIT license, just don't repackage it and call it your own please!
Also if you do make some changes, feel free to make a pull request and help make things more awesome!

