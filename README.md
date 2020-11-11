<div align="center"> 
<img src="https://images.unsplash.com/photo-1562054438-f789d60d03eb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=500&q=60" width="700px;">
<h1>My first Deep Learning Project with PyTorch<h1>
<h2>Classification of DogBreeds<h2>
</div>

# Motivation

As my final Capstone Project at the Udacity Nanodegreeprogramm I tried to classify dogbreeds. Therefore I used the Deep Learning Library PyTorch. 

# Results

With the accuracy metric I could show that even with a non pre-trainied moddel I got an accuracy of 60%. In order to get better results I used different Deep Learning models and could finally acieve an accuracy of arround 80%.

## Content

- Python Notebook as [iPython](https://github.com/bohniti/capstone-project/blob/master/dog-breed-classifier-final.ipynb)
- Report as [Medium Post](https://timo-bohnstedt.medium.com/my-first-deep-learning-project-with-pytorch-dogbreeds-5750fbc0f9da)


## Structure

```                     
|  Dog_Breed_Classifier-Final.ipynb
|   +-- Input                      
|         +-- Your unpackt Kaggle dataset here                       
|   |                  
+-- environment.yml                    
+-- readme.md
+-- .gitignore    

```

## Prerequisites

Create the conda environment in order to see every library which I had used to ensure I could handle the data and finally train the network. 

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

