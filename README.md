# Classification-of-Texts---Naive-Bayes-Classification

Environment: Python 3.6
IDE used : Pycharm
Libraries used: nltk.corpus, sklearn.metrics, re, numpy

1. Problem Sections:
• Loading Netnews text articles Dataset and reading it
The Iris dataset is taken from the http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naivebayes.html repository.
We read each folder of the dataset and each files in the subfolders for the text messages. Considering random shuffling of 500 msgs from each class folders i.e., 10000 messages we constitute a training_set. Similarly, we consider the
randomly shuffled 10000 messages as testing_set.

2. Data:
A dataset containing 20,000 newsgroup messages drawn from the 20 newsgroups. The dataset contains 1000 documents from each of the 20 newsgroups and this is downloaded from: http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naivebayes.html (Newsgroup Data)

3. Method:
model_fit () :method for creating the training set from the dictionary comprising of words and frequence of words in the feature set of each class from 20 newsgroup dataset
probability_log(): method for calculating the individual probability of each test file in a particular class
ind_probability(): method which helps the prediction method in calculating the prediction values of one testing file at a time
predictions (): method to predict the class label from testing set using the model_fit method
new_possibility(): method for calculating the Naïve Bayesian log probability for each testing set element from a class

4. Results:
Accuracy score of Naïve Bayes classifier is the output and is found to be: 0.05
Also, classification report is jotted down using the sklearn library with testing and predicted class dataset comparisons. That is the report consisting of precision, recall, f1-score and support values are plotted against each class folder names with average/total value printed at the end of the report.


References:
https://stackoverflow.com/questions/16819222/how-to-return-dictionary-keys-as-a-list-in-python
https://www.saltycrane.com/blog/2007/09/how-to-sort-python-dictionary-by-keys/
https://docs.quantifiedcode.com/python-antipatterns/readability/not_using_items_to_iterate_over_a_dictionary.html
https://machinelearningmastery.com/clean-text-machine-learning-python/
