import pandas as pd
import numpy as np
import scipy as sp
from scipy.special import expit as sigmoid_function
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')


def load_data(location):
    """ Given a directory string, returns a pandas dataframe containing hw data."""
    
    # dictionary containing various matrices and some metadata
    data = sp.io.loadmat(location)
    x = pd.DataFrame(data['X'])
    y = pd.DataFrame(data['y'], columns=['digit_class'])
    y[y == 10] = 0 # convert from matlab's 1-index to python's 0-index
    return x, y


def visualize_digit_images_data(data, gridSize=(10, 10), desiredDigitIndices=None, title=None):
    """ Provides a plot of image data so we can see what we are playing with. 

    The kwarg allows for the option of hand selecting digit images we desired to see. """
    
    # thanks to pdf we know data is (5000,400). for plotting images, we want to 
    # take it to (5000,20,20)
    pixelSquares = pd.Panel(data.values.reshape(5000, 20, 20)).transpose(0, 2, 1)

    # we have to manually build the image by stitching together individual digits
    # first, we choose the digits we want
    if desiredDigitIndices is None:
        desiredDigitIndices = []

    desiredDigits = pixelSquares.ix[desiredDigitIndices, :, :] # for default kwarg, this is empty
    randomDigits = pixelSquares.sample(gridSize[0] * gridSize[1] - len(desiredDigitIndices), axis=0) # get remaining images 
    allDigits = pd.concat([desiredDigits, randomDigits], axis=0)

    # now we must fill in the matrix that represents the picture
    pixelRows = 20 * gridSize[0]
    pixelCols = 20 * gridSize[1]
    digitImage = np.zeros((pixelRows, pixelCols))
    digitToPlot = -1 

    for i in range(0, pixelRows, 20):
        for j in range(0, pixelCols, 20):

            digitToPlot += 1
            digitImage[i:i+20, j:j+20] = allDigits.iloc[digitToPlot]

    # lastly we convert to Pillow image (accepted by mpl) and plot
    digitImage = sp.misc.toimage(digitImage)
    plt.figure()
    plt.imshow(digitImage, cmap=mpl.cm.Greys)

    if title is None:
        title = ''
    plt.title(title)
    return 


# shamelessly stolen from my own hw2, where i had previously written this
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


def train_one_vs_all(features, response, classes, regularizationParameter, numIters=500):
    """ Trains classifiers for the provided number of classes, returning optimal model parameters 
    in a len(classes) x numFeatures parameter matrix. """
    
    # some preprocessing
    features.insert(0, 'intercept', 1)
    optimalTheta = np.zeros((len(classes), len(features.columns))) 

    # as specified by the hw, train separate models for the classes
    for model in range(len(classes)):

        print('Training model {0}'.format(classes[model]))

        classResponse = pd.get_dummies(response, columns=['digit_class'])['digit_class_' + str(classes[model])] # what a great func
        res = sp.optimize.minimize(compute_cost, 
                                   np.zeros(len(features.columns)), # initial theta (1 x 401)
                                   args=(features, classResponse, regularizationParameter),
                                   jac=compute_dCost_dTheta,
                                   options={'maxiter': numIters},
                                   method='CG')
        optimalTheta[model,:] = res.x

    return optimalTheta


def predict_one_vs_all(features, optimalTheta, numClasses):
    """ Returns a len(features)-vector of predicted class labels, given an optimalTheta calculated
    from a training routine. 

    The numClasses parameter is required as a sanity check. It is possible that the user expects more or fewer
    classes than were passed into the training routine, in which case this model would be ill-defined. """
    
    assert optimalTheta.shape[0] == numClasses, 'The passed number of classes is not the same as was used in training.'
    return np.argmax(np.dot(features, optimalTheta.T), axis=1)


def logistic_regression_main(dataLoc):
    """ Part 1 of the homework """
    
    digitFeatures, digitResponse = load_data(dataLoc)
    optimalTheta = train_one_vs_all(digitFeatures, digitResponse, np.arange(0, 10, 1), 0.1)
    predictions = predict_one_vs_all(digitFeatures, optimalTheta, 10)

    digitResponse['logistic_regression_predictions'] = predictions
    accuracy = digitResponse['digit_class'] - predictions
    accuracy = accuracy.value_counts() / len(digitResponse)
    print('Logistic regression accuracy: {0}'.format(accuracy))

    digitFeatures.drop('intercept', axis=1, inplace=True) # remove the added column 

    
    for i in np.random.randint(0, 5001, 10):

        titleStr = 'Predicted: {0}, Actual: {1}'.format(digitResponse.ix[i, 'logistic_regression_predictions'], 
                                                        digitResponse.ix[i, 'digit_class'])
        visualize_digit_images_data(digitFeatures, gridSize=(1,1), 
                                    desiredDigitIndices=[i], 
                                    title=titleStr)
    return 




def load_neural_network_weights(location):
    """ DocString"""
    
    data = sp.io.loadmat(location)
    thetaOne = pd.DataFrame(data['Theta1'])
    thetaTwo = pd.DataFrame(data['Theta2'])
    return thetaOne, thetaTwo


def feed_forward_propagate_and_predict(features, layerParameters):
    """ DocString"""
    
    features = features.values # change pandas dataframe to np array

    for layer in layerParameters:

        features = np.insert(features, 0, 1, axis=1)
        z = np.dot(features, layer.T)
        features = sigmoid_function(z)

    return np.argmax(features, axis=1)



def neural_networks_main(dataLoc, weightLoc):
    """ Part 2 of the homework """
    
    layerParams = load_neural_network_weights(weightLoc)
    digitFeatures, digitResponse = load_data(dataLoc)

    predictions = feed_forward_propagate_and_predict(digitFeatures, layerParams)
    predictions = np.arange(1, 11, 1)[predictions] 
    predictions[predictions == 10] = 0
    digitResponse['network_predictions'] = predictions # must convert back to the 0-indexing

    accuracy = digitResponse['digit_class'] - digitResponse['network_predictions']
    accuracy = accuracy.value_counts().ix[0] / len(digitResponse)
    print('Neural network accuracy: {0}'.format(accuracy))

    for i in np.random.randint(0, 5001, 10):

        titleStr = 'Predicted: {0}, Actual: {1}'.format(digitResponse.ix[i, 'network_predictions'], digitResponse.ix[i, 'digit_class'])
        visualize_digit_images_data(digitFeatures, gridSize=(1,1), 
                                    desiredDigitIndices=[i], 
                                    title=titleStr)
    return 

if __name__ == '__main__':
    
    dataLocation = r"C:\Users\ashamlian\Downloads\machine-learning-ex3\ex3\ex3data1.mat"
    weightLocation = r"C:\Users\ashamlian\Downloads\machine-learning-ex3\ex3\ex3weights.mat"

    logistic_regression_main(dataLocation)
    neural_networks_main(dataLocation, weightLocation)