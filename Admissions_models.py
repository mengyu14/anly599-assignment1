import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

###########################################################################
###### Logistic Regression
###########################################################################
def LRegression(X, y, k):
    """
    This function takes in the target variable and the independent variables and fits the
    logisitic regression model using k-fold cross validation.

    :param X: the dataframe containing the independent variables
    :param y: the target variable
    :param k: number of folds determined by the user
    :return: list of k accuracy scores
    """
    X = pd.get_dummies(X, drop_first=False)
    lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=200)
    scores = cross_validate(lr, X, y, cv = k, return_estimator=True)

    """
    # prints model coefficients
    for model in scores['estimator']:
       print(model.coef_)
    """

    return scores['test_score']


###########################################################################
###### Decision Trees
###########################################################################
def DTree(X, y, k):
    """
    This function takes in the target variable and the independent variables and fits the
    decision tree model using k-fold cross validation.

    :param X: the dataframe containing the independent variables
    :param y: the target variable
    :param k: number of folds determined by the user
    :return: list of k accuracy scores
    """
    # create dummy variables from categorical data
    X = pd.get_dummies(X, drop_first=False)
    # define model and fit data
    dtree = DecisionTreeClassifier()
    scores = cross_validate(dtree, X, y, cv = k, return_estimator=True)

    """
    # print and save each graph into pdf
    for i, trees in enumerate(scores['estimator']):
        dot_data = tree.export_graphviz(trees, out_file=None,
                                        feature_names=X.columns)
        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)

        # Show graph
        Image(graph.create_png())

        name = "DT"+str(i)+".pdf"
        # Create PDF
        graph.write_pdf(name)
    """
    return scores['test_score']


###########################################################################
###### Random Forest
###########################################################################
def RForest(X, y, k):
    """
    This function takes in the target variable and the independent variables and fits the
    random forest model using k-fold cross validation.

    :param X: the dataframe containing the independent variables
    :param y: the target variable
    :param k: number of folds determined by the user
    :return: list of k accuracy scores
    """
    # create dummy variables from categorical data
    X = pd.get_dummies(X, drop_first=False)
    # define model and fit data
    rForest = RandomForestClassifier(n_estimators=1000)
    scores = cross_validate(rForest, X, y, cv = k, return_estimator=True)

    """ Feature Importance Plot
    col = list(X.columns)
    for i, forest in enumerate(scores['estimator']):
        y = forest.feature_importances_
        # feature importance plot
        fig, ax = plt.subplots()
        width = 0.4  # the width of the bars
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, width, color="green")
        ax.set_yticks(ind + width / 10)
        ax.set_yticklabels(col, minor=False)
        name = 'Feature importance in RandomForest Classifier'+str(i)
        plt.title(name)
        plt.xlabel('Relative importance')
        plt.ylabel('feature')
        plt.figure(figsize=(5, 5))
        fig.set_size_inches(6.5, 4.5, forward=True)
        #plt.savefig(name)
        #plt.show()
    """
    return scores['test_score']

###########################################################################
###### Naive Bayes
###########################################################################
def NBayes(X, y, k):
    """
    This function takes in the target variable and the independent variables and fits the
    multinomial naive bayes model using k-fold cross validation.

    :param X: the dataframe containing the independent variables
    :param y: the target variable
    :param k: number of folds determined by the user
    :return: list of k accuracy scores
    """

    # create dummy variables from categorical data
    X = pd.get_dummies(X, drop_first = False)
    # define model and fit data
    nBayes= MultinomialNB()
    scores = cross_validate(nBayes, X, y, cv=k, return_estimator=True)

    return scores['test_score']


###########################################################################
###### SVM
###########################################################################

def SVM(X, y, k):
    """
    This function takes in the target variable and the independent variables and fits the
    SVM model using k-fold cross validation. The function loops through three SVM kernels as
    well as 10 cost parameters and prints our the best model with the highest accuracy score
    for each kernel.

    :param X: the dataframe containing the independent variables
    :param y: the target variable
    :param k: number of folds determined by the user
    :return: print the best model for each of the SVM kernels
    """

    C = np.arange(0.1, 1.05, 0.1) # SVM regularization parameter
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)
    # loop through three different models and 10 different parameter values
    for kernel in ('linear', 'poly', 'rbf'):
        acc_score = 0
        parameter = 0
        for c in C:
            svmMod = svm.SVC(kernel=kernel, C=c)
            scores = cross_validate(svmMod, X, y, cv=k, return_estimator=True)
            avgAcc = sum(scores['test_score']/len(scores['test_score']))

            # find out highest accuracy score and store the parameter as well as the results
            if(avgAcc>acc_score):
                parameter = c
                acc_score = avgAcc

        # print the results for easier reading
        print(f"The SVM model with {kernel} kernel has the best accuracy with cost parameter C = {parameter:0.01f} and average accuracy score of {acc_score:0.03f}")

def main(argv):
    admin = pd.read_csv('Admissions_Cleaned.csv',sep=',', encoding='latin1')
    #print(admin.head)

    # change gender and Volunteer levels into categorical variables
    admin['Gender'] = admin['Gender'].astype(str)
    admin['VolunteerLevel'] = admin['VolunteerLevel'].astype(str)

    # Define target variable and independent variables
    y = admin['Decision']
    X = admin.drop(columns = ['Decision'])

    # Decision Tree models
    dtree = DTree(X, y, 5)
    print(f"The average accuracy score for decision tree model is {sum(dtree)/len(dtree):0.03f}")

    # Random Forest
    rforest = RForest(X, y, 5)
    print(f"The average accuracy score for random forest is {sum(rforest)/len(rforest):0.03f}")

    # Naive Bayes
    nbayes = NBayes(X, y, 5)
    print(f"The average accuracy score for multinomial naive bayes is {sum(nbayes)/len(nbayes):0.03f}")

    # SVM
    SVM(X, y, 5)

    # Logistic Regression
    logfit = LRegression(X, y, 5)
    print(f"The average accuracy score for logistic regression is {sum(logfit)/len(logfit):0.03f}")


if __name__ == "__main__":
    main(sys.argv)
