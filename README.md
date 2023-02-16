
# Graded assignment 1 - text classification using Genetic Algorithms

## Introduction

In this assignment we'll make a binary text classifier using neural networks and train it using genetic algorithms. We will classify movie reviews from IMDB as either negative or positive.

The assignment text contain 3 steps, so I divided the assignment in 3 sections:

1.  Preprocessing of the text
2.  Genetich Algorithm
3.  Testing

**Preprocessing text**

First I uploaded all data and put them together. Then I explore the dataset, explore the use of vocabulary distribution, review length distribution.

After analyzing data I split them into training, validation and testing set. Then I preprocessed the text using different methods like removing stop words, lemmatization etc.

Training set will be used to build vocabulary and IDF values. Validation set will be used when assigning fitness in Genetic Algorithm which I'll get into more. Testing set will be used at the end in validation part for a final evaluation.

What we basically doing here is to try to predict what class a text belongs to based on how many of which words the text contains. Our classifier must recognize that some words convey positive sentiment, some convey negative sentiment and some make no difference. The latter words don't help with predictions. That's why we do preprocessing and turn all the reviews into equally sized vectors representing the frequency and importance of each word which can be proceeded by our neural network. How we do the preprocessing will be further explained later.

I have built two classes for preprocessing stage. One is TextPreprocessor which processes contents with methods like removing stop words, lemmatization etc.

The other is TF-IDF vectorizer class which is used to build a Numpy array of TF-IDF vectors. Each document is turned into a TF-IDF vector.

**Genetic Algorithm**  Genetic algorithm (often referred as GA) is an algorithm inspired by evolution theory. Here rather than species we have solutions to a problem. We call those solutions for Chromosomes. In this case Chromosomes are just weights for a neural network. We use methods like "cross-over" and "mutation" to make changes in those Chromosomes, and we try to pick the best Chromosome. In here the best Chromosome is

The classifier I used is a simple neural network. I considered decision tree, but I'm not familiar with decision trees while I'm a bit familiar with neural network. Thus I decided to go with neural network.

The neural net has only 2 hidden layers, 16 and 8 hidden nodes. I kept it simple because I had to take hardware limitations into account too. In early trials, this architecture seemed more than capable of learning from the training set. In the Genetic Algorithm I will train with the trainign dataset but I will calculate fitness using the validation set for each classifier. This will avoid overfitting the training set.

**Testing**  The validation data is simply using the optimizied weights to predict all reviews in the testing set to see how our GA performs - that is, to determine its fitness. In the assignment text they suggested to use testing set for assigning fitness in GA and in validation part. But I thought that using the same data for GA improvement and evaluation would led to overfitting. Thus, other than training set, I used two separate sets: validation set and testing set. Validation set will be used when assigning fitness while optimizing weights in GA. Testing set will be used at the end.
