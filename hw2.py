from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import scipy as sp



# Logistic Regression
def load_data(fileLocation, featureColumns, responseColumns):
    """ Given a file location, this provides a pandas DataFrame of the data.
    featureColumns and responseColumns are lists of indices corresponding to those data series,
    with the convention that the returned data is from left to right, features and then responses. """
    return pd.read_csv(fileLocation, header=None, usecols=featureColumns + responseColumns)


def plot_data(data):
    """ Returns a scatter plot that visualizes the passed dataframe. Currently, tailored to merely 
    encapsulate very specific visualization. """
    
    # lets play with namedtuples for fun. kinda like a struct-ish
    PlotArgs = namedtuple('PlotArgs', ['color', 'label', 'marker'])
    plotting = {0: PlotArgs('RoyalBlue', '0', 'x'), 1: PlotArgs('GoldenRod', '1', 'o')}

    data.columns = ['exam_1', 'exam_2', 'admission']

    # look at how neat the namedtuple is!
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    for adminStat, grouped in data.groupby('admission'):
        adminStatPlotConfig = plotting[adminStat]
        grouped.plot.scatter(x='exam_1', y='exam_2', ax=ax,
                             color=adminStatPlotConfig.color, 
                             label=adminStatPlotConfig.label, 
                             marker=adminStatPlotConfig.marker)

    return ax


def sigmoid_function(data):
    """ Returns the sigmoid 1 / (1 + e^-x), returned in the same shape as X"""
    
    # # this can be done with
    # import scipy as sp
    # sp.special.expit(x) # for performance and no translation
    # # or 
    # sp.stats.logistic.cdf(x) # wrapper that has many other operations available

    return 1 / (1 + np.exp(-data))


def compute_cost(theta, features, response, regularizationParameter=0):
    """ Returns the logistic regression func, evaluated on the data set and passed theta. This 
    also provides the opportunity for regularization.  """

    # set up regularization so that we always ignore the intercept parameter
    interceptKnockOut = np.ones(len(features.columns))
    interceptKnockOut[0] = 0
    
    regularization = np.dot(interceptKnockOut, theta**2) # this is SUM (i=1, numFeatures) theta_i^2
    regularization = regularization * regularizationParameter / (2 * len(features))

    features = np.dot(theta, features.T) # dont forget H(x; theta) = sigmoid(innerprod(theta, features))

    # build up the cost function one step at a time
    cost = sigmoid_function(features)
    cost = response * np.log(cost) + (1 - response) * np.log(1 - cost)
    cost = -cost.sum(axis=0) / len(features)
    return cost + regularization


def compute_dCost_dTheta(theta, features, response, regularizationParameter=0):
    """ Returns the gradient of the cost function with respect to theta, evaluated on the data """
    
    # set up regularization so that we always ignore the intercept parameter
    interceptKnockOut = np.ones(len(features.columns))
    interceptKnockOut[0] = 0

    regularization = interceptKnockOut * theta # no summation this time, so just elementwise mult
    regularization = regularization * regularizationParameter / len(features)

    dottedFeats = np.dot(theta, features.T)

    gradient = sigmoid_function(dottedFeats) - response
    gradient = gradient[:,np.newaxis] * features 
    gradient = gradient.sum(axis=0)
    return gradient / len(features) + regularization


def plot_decision_boundary(data, optimalTheta):
    """ Returns a scatter plot with the optimized decision boundary plotted """
    
    scatterAx = plot_data(data)

    # in order to plot the decision boundary we need to convert the regression into a new equation
    convertedDecisionBoundary = lambda x: -optimalTheta[1] * x / optimalTheta[2] - optimalTheta[0] / optimalTheta[1]
    # remember that data does not contain intercept column here, hence the 0 index
    scatterAx.plot(data.iloc[:,0], convertedDecisionBoundary(data.iloc[:,0]), 'k-') 
    return 


def predict_student_admission_probability(examOneScore, examTwoScore, optimalTheta, probability=False):
    """ Retuns the admission likelihood of a student with the given exam scores. By passing a boolean to 
    probability, the function will return the class or the probability (default False) """

    # one student case
    if isinstance(examOneScore, int):
        features = np.array([1, examOneScore, examTwoScore])

    # many students at once
    elif isinstance(examOneScore, pd.DataFrame) or isinstance(examOneScore, pd.Series):
        features = pd.concat([examOneScore, examTwoScore], axis=1)
        features.insert(0, 'intercept', 1)

    else:
        raise TypeError("examOneScore and examTwoScore must be int's or pandas DataFrame or Series objects")

    transformedFeatures = np.dot(optimalTheta, features.T)
    responseHat = sigmoid_function(transformedFeatures)
    return responseHat if probability else np.round(responseHat)


# Regularized Logistic Regression
def _map_feature(X1, X2, degree):
    """ Creates additional polynomial features up to specified degree"""
    
    # set up a new index for either iterables or numeric types
    try:
        idx = range(len(X1))
    except TypeError:
        idx = [1]

    mapped = pd.DataFrame(index=idx)
    count = 1 # for col names

    for i in range(1, degree+1):
        for j in range(0, i + 1):

            mapped['feature_' + str(count)] = X1**(i-j) * X2**j
            count += 1

    return mapped


def plot_regularized_decision_boundary(data, optimalTheta):
    """ Returns a scatter plot with a fitted decision boundary overlain. This is slow as hell. """
    
    scatterAx = plot_data(data.iloc[:, [1, 2, -1]]) # two real features and response

    # this time we cant calculate the decision boundary analytically
    # we will meshgrid feature 1 and 2 and then draw a contour plot
    fauxFeatOne = np.linspace(-1, 1.5, 50)
    fauxFeatTwo = np.linspace(-1, 1.5, 50)
    pointsInSpace = np.meshgrid(fauxFeatOne, fauxFeatTwo)
    predictionsAsZAxis = np.zeros((len(fauxFeatOne), len(fauxFeatTwo)))

    for i in range(len(fauxFeatOne)):
        for j in range(len(fauxFeatOne)):

            feats = _map_feature(fauxFeatOne[i], fauxFeatTwo[j], 6)
            feats.insert(0, 'intercept', 1)

            # we are going to hack the predict_student... function to work with larger pandas objs
            rowPredictions = predict_student_admission_probability(feats.iloc[:,1], feats.iloc[:,2:], optimalTheta)
            predictionsAsZAxis[i, j] = rowPredictions

    plt.contour(pointsInSpace[0], pointsInSpace[1], predictionsAsZAxis, levels=[.5])
    return 



def regularized_log_reg_main(fileLocation):
    """ Section 2 of teh homework"""

    data = load_data(fileLocation)
    # plot_data will work for this data set as well (2 features and one binary response variable)
    plot_data(data)

    # formatting and initializing features and params 
    microChipFeatures = _map_feature(data.iloc[:,0], data.iloc[:,1], 6)
    microChipFeatures.insert(0, 'intercept', 1)
    microChipResponse = data.iloc[:, -1]
    initialTheta = np.zeros(len(microChipFeatures.columns))
    regularizationParameter = 1

    print('The initial theta cost is {0}'.format(compute_cost(initialTheta, microChipFeatures, microChipResponse, regularizationParameter)))
    print()

    res = sp.optimize.minimize(compute_cost, initialTheta, 
                               args=(microChipFeatures, microChipResponse, regularizationParameter), 
                               jac=compute_dCost_dTheta,
                               method='BFGS')
    optimalTheta = res.x

    print('The final theta cost is {0}'.format(compute_cost(optimalTheta, microChipFeatures, microChipResponse, regularizationParameter)))
    print()

    # one can vary the regularization parameter, optimize and plot decision boundaries to see the effect
    # i omit it here because its a) slow, and b) changes to one line 
    # we concat because the true features were mapped to new ones
    plot_regularized_decision_boundary(pd.concat([microChipFeatures, microChipResponse], axis=1), optimalTheta)
    
    # here is an in sample accuracy, with the same hack as in plot_regularized_decision_boundary()
    trainingPredictions = predict_student_admission_probability(microChipFeatures.iloc[:,1], microChipFeatures.iloc[:,2:], optimalTheta)
    inSampleAccuracy = trainingPredictions - microChipResponse
    inSampleAccuracy = inSampleAccuracy.value_counts().ix[0] / len(data)
    print('The in-sample accuracy of the model with lambda={0} is {1}'.format(regularizationParameter, inSampleAccuracy))



def logistic_regression_main(fileLocation):
    """ Section 1 of the homework """

    # load and visualize for the first time
    data = load_data(fileLocation, [0,1], [2])
    plot_data(data)

    # format the data for the optimization steps
    studentFeatures, studentResponse = data.iloc[:,:-1], data.iloc[:,-1]
    studentFeatures.insert(0, 'intercept', 1)
    initialTheta = np.array([0,0,0])

    print('The initial theta cost is {0}'.format(compute_cost(initialTheta, studentFeatures, studentResponse)))
    print()

    res = sp.optimize.minimize(compute_cost, initialTheta, args=(studentFeatures, studentResponse), jac=compute_dCost_dTheta, method='BFGS')
    optimalTheta = res.x

    print('The final theta cost is {0}'.format(compute_cost(optimalTheta, studentFeatures, studentResponse)))
    print()

    # visualize the data with a boundary after optimization
    plot_decision_boundary(data, optimalTheta)

    # correct predictions will be 0 - 0 or 1 - 1 = 0
    trainingPredictions = predict_student_admission_probability(studentFeatures.iloc[:,0],studentFeatures.iloc[:,1], optimalTheta)
    inSampleAccuracy = trainingPredictions - studentResponse
    inSampleAccuracy = inSampleAccuracy.value_counts().ix[0] / len(data)
    print('The model has an in-sample accuracy rate of {0}'.format(inSampleAccuracy))

    return


if __name__ == '__main__':
    
    logistic_regression_main(r"a;dsfasd;lfj")
    regularized_log_reg_main(r"a;ldfa;lsdfj")