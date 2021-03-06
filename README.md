# HuffPost Dataset

This is a Data Science project for Exploratory Data Analysis and Machine Learning modeling using the Huffpost dataset and Python. 

The dataset can be downloded from the following link ([link to Dataset](https://www.kaggle.com/rmisra/news-category-dataset)). This dataset has around 200k huffPost articles from 2012 to 2018. It contains headlines, short descriptions, authors and category for each article.

![Frequency by Category](./imgs/FreqByCategory.PNG)

If you want to reproduce the code in this repository, please download and extract json in the following path "./Data/News_Category_Dataset_v2.json" before running the notebooks (NOTE: Given the size of the datasets, some models take long times to complete).

The programming language used is Python with [Altair](https://altair-viz.github.io/) visualization library on Jupyter Notebooks. Since almost all Altair plots are interactive and Github does not render Altair plots correctly, I recommend that the html file is downloaded and rendered in a browsers, such as Chrome or Firefox, so that you are able to interact with the plots (Github only renders plain Matplotlib plots). For this reason, each jupyter file has a html file with the same name. 

Here is one of the examples of Altair plots which was made for exploring the Neural Network accuracy for high-dimensional hyperparameter search (You can find the interactive version of the following [here](https://github.com/sotomsa/HuffPostDataSet/blob/main/News_Headlines_to_Category_Model.html)).

![Neural Network High Dimensional Parameter Space Visualization](./imgs/NN_parameter_space.PNG)

What is interesting about this interactive plot is that it allows you to select a rectangular area in one of the plots and being able to see the position of the enclosed points in other dimensions of the parameter space. For example, in the first plot to the left, the best performance runs and its running times are selected, while in the other 2 plots to the right one can see another 4 parameter dimensions of the same runs. With this tool is easy to see that the best performance runs are those with a number of epochs higher than 15, a learning rate between (2e-5, 3.5e-3) and a dropdown close to 0.5 independent of the batch_size.

## Contents

Here is the list of files and their descriptions:

### Supervised Machine Learning Models

Machine Learning models for classifying "headlines" or "headline + short description" as one of the 41 categories in the dataset (or 7 categories simplified version). Here is the link to the jupyter notebook [News_Headlines_to_Category_Model.ipynb](https://github.com/sotomsa/HuffPostDataSet/blob/main/News_Headlines_to_Category_Model.ipynb) and the link to its html file [News_Headlines_to_Category_Model.html](https://github.com/sotomsa/HuffPostDataSet/blob/main/News_Headlines_to_Category_Model.html). Some of the highlights included in this jupyter notebook include:

- Models such as Naive Bayes (NB), Logistic Regresion (LR) with/without PCA , Random Forests (RF) and Neural Networks (NN) (Dense FeedForward and LSTM with Glove 100 word embeddings and Pretrained model). It uses libraries sucha as Sklearn, Keras and Pythorch.
- Grid Search (sklearn) and Bayesian Optimization (skopt) for hyperparameter search.
- Advanced interactive plots with Declarative Visualization using Altair.
- Conclusions section with some high-level comparison among the models.

Here are some results for the models tested:

![Accuracy of the Models](./imgs/results.PNG)


| model_name                            |   train_accuracy |   test_accuracy |
|:--------------------------------------|-----------------:|----------------:|
| headlines_description_7_categories_LR |             0.69 |            0.74 |
| headlines_description_7_categories_NB |             0.65 |            0.68 |
| headlines_pytorch_pretrained_NN       |             0.62 |            0.63 |
| headlines_elasticnet_LR               |             0.59 |            0.59 |
| headlines_LSTM_NN                     |             0.65 |            0.55 |
| headlines_description_NB              |             0.51 |            0.53 |
| headlines_RF                          |             0.51 |            0.52 |
| headlines_ngrams_NB                   |             0.48 |            0.51 |
| headlines_NB                          |             0.48 |            0.5  |
| headlines_NN                          |             0.46 |            0.47 |
| short_description_NB                  |             0.37 |            0.39 |
| headlines_PCA_LR                      |             0.37 |            0.38 |

### Unsupervised Machine Learning Model

In this second notebook called [Authors.ipynb](https://github.com/sotomsa/HuffPostDataSet/blob/main/Authors.ipynb) deals with a clustering algorithm for finding patterns in the categories of articles produced by different authors. In particular, it tries to find different group of authors with similar writting patterns. The html version can be found in the link [Authors.html](https://github.com/sotomsa/HuffPostDataSet/blob/main/News_Headlines_to_Category_Model.html).

In the following image, you can see that the algorithm found 4 clusters:

- Cluster 0: cluster with authors writing mainly about politics.
- Cluster 1: cluster with authors writing about wellness, healthy living and parenting.
- Cluster 2: cluster with authors writing almost only about travelling. This is the smalles cluster.
- Cluster 3: This is definetely the largest cluster of all with authors that seem not to fall in any previous clusters.

![Categories by Cluster](./imgs/categories_freq_by_cluster.PNG)

In the following image you can see the frequency of authors by cluster:

![Frequency by Cluster](./imgs/freq_by_cluster.PNG)