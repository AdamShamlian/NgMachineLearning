import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# warmup_exercise: 
def warmup_exercise():
    """ Return a 5x5 identity matrix. """
    return np.eye(5)


# Linear Regression with One Variable
def load_data(fileLocation, featureColumns, responseColumns):
    """ Given a file location, this provides a pandas DataFrame of the data.

    featureColumns and responseColumns are lists of indices corresponding to those data series,
    with the convention that the returned data is from left to right, features and then responses. """
    return pd.read_csv(fileLocation, header=None).iloc[:, featureColumns + responseColumns]


def plot_scatter(data):
    """ Generates a scatter plot of the provided data. Returns the axes for further plotting ability. """

    data.columns = ['city_population', 'profit']
    axes = data.plot(x='city_population', y='profit', kind='scatter')
    axes.set_xlim(left=data['city_population'].min() - 1)
    axes.set_xlabel('Population of city in 10,000s')
    axes.set_ylabel('Profit in $10,000s')
    return axes
        

def compute_cost(features, response, theta):
    """ Computes the least square error cost function value. """
    
    # in this case we know that the last variable is the response, so we ignore it
    cost = np.dot(features, theta) # h_theta(x_i)
    cost -= response # - y_i
    cost *= cost # square it
    cost = cost.sum() / (2 * len(features)) # summation, then divide by 2m
    return cost


def compute_dCost_dTheta(features, response, theta):
    """ Helper function that calculates dJ/dTheta, evaluated at theta. """
    
    # notice that this will be similar to the general cost function
    cost = np.dot(features, theta) # h_theta(x_i)
    cost -= response # - y_i
    cost = np.broadcast_to(cost, (len(features.columns), len(cost))).T * features # elementwise mult by the remainder of derivative
    cost = np.sum(cost, axis=0) / len(features)
    return cost.values


def execute_gradient_descent(data, learningRate=0.01, maxIters=2000, theta=None, addInterceptFeature=True):
    """ Returns the parameter vector Theta of len(data.columns) + 1 that minimizes least
    squares error loss. 

    addInterceptFeature - if True, adds a feature column of 1's corresponding to the Theta_0 parameter
    theta - option to initialize Theta to specific values. 
    learningRate - algorithm learning rate parameter
    maxIters - number of iterations after which the algorithm automatically terminates """
    
    # initialize theta if not given
    if theta is None:
        interceptLength = 0 if addInterceptFeature else -1
        theta = np.zeros(len(data.columns) + interceptLength)

    # insert intercept feature 
    data.insert(0, 'intercept', 1)

    # print initial cost, and set up a cost tracking list
    cost = compute_cost(data.iloc[:,:-1], data.iloc[:,-1], theta)
    costProgression = [cost]
    print('Current _cost: {0}'.format(cost))

    for i in range(maxIters):

        theta -= learningRate * compute_dCost_dTheta(data.iloc[:,:-1], data.iloc[:,-1], theta) # dCost_dTheta is a 2-vector
        
        cost = compute_cost(data.iloc[:,:-1], data.iloc[:,-1], theta)
        costProgression.append(cost)
        print('Current cost: {0}'.format(cost))

    # remove the dummy intercept variable from data, else our scatter plots will be messed up
    data.drop('intercept', axis=1, inplace=True)
    return theta, pd.DataFrame(costProgression)


def plot_fitted_regression(data, fittedTheta):
    """ Plots the fitted regression line on a previous scatter plot. """
    
    scatter = plot_scatter(data)

    x = np.arange(data['city_population'].min(), data['city_population'].max(), .1) # recall the name of our columns
    scatter.plot(x, fittedTheta[0] + fittedTheta[1] * x)

    scatter.legend(['Training data', 'Linear Regression'])
    return 


def predict(observationVector, theta):
    """ Given a fitted theta and observation vector, we can make a regression prediction. 
    We assume observationVector and theta are lists. """
    
    observationVector.insert(0, 1) # add the intercept value
    return sum([observationVector[i] * theta[i] for i in range(len(theta))])


def plot_cost_curves(data):
    """ Plots 3d and contour curves of the cost function in parameter space. This helps us
    visualize the dynamics of the gradient descent algorithm. """

    # initialize our parameter space
    thetaZeroValues = np.linspace(-10, 10, 100)
    thetaOneValues = np.linspace(-1, 4, 100)
    costValues = np.zeros((100, 100))

    # we are going to need to insert the intercept column again
    data.insert(0, 'intercept', 1)

    # get the fuction values over parameter space
    for i in range(100):
        for j in range(100):
            costValues[i,j] = compute_cost(data.iloc[:,:-1], data.iloc[:,-1], [thetaZeroValues[i], thetaOneValues[j]])

    # drop intercept
    data.drop('intercept', axis=1, inplace=True)

    # convert to 3d friendly meshgrid
    thetaZeroValues, thetaOneValues = np.meshgrid(thetaZeroValues, thetaOneValues)

    # time to plot, first the surface and then the contour
    fig = plt.figure(figsize=(15,10))
    
    # surface plot
    surfax = fig.add_subplot(1, 2, 1, projection='3d')
    surfax.plot_surface(thetaZeroValues, thetaOneValues, costValues.T)

    # contour plot
    contax = fig.add_subplot(1, 2, 2)
    contax.contour(thetaZeroValues, thetaOneValues, costValues.T, 17) # seventeen contour lines for clarity
    contax.set_ylabel('Theta One')
    contax.set_xlabel('Theta Zero')
    return 


def required_main():
    """ Main function for required exercises """

    loc = r"C:\Users\ashamlian\Downloads\machine-learning-ex1\ex1\ex1data1.txt"
    foodTruckData = load_data(loc, [0], [1])
    plot_scatter(foodTruckData)

    fittedTheta, _ = execute_gradient_descent(foodTruckData, theta=[0,0])

    print('For a town of {0} people, we expect profits of {1} * $10,000'.format(35000, predict([3.5], fittedTheta)))
    print('For a town of {0} people, we expect profits of {1} * $10,000'.format(70000, predict([7], fittedTheta)))

    plot_fitted_regression(foodTruckData, fittedTheta)
    plot_cost_curves(foodTruckData)
    return 




# Optional Exercises
def normalize_features(data):
    """ Apply standardization to the dataset. Pandas makes this very simple """
    
    data.columns = ['sq_ft', 'bedrooms', 'price']
    tmp = data[['sq_ft', 'bedrooms']]
    tmp = tmp.sub(tmp.mean()).div(tmp.std()) 
    return pd.concat([tmp, data['price']], axis=1) 


def plot_convergence(costProgression, learningRate):
    """ Constructs a plot that demonstrates the convergence rate for a given learning rate. """
    
    x = np.arange(1, len(costProgression)+1)

    fig, ax = plt.subplots(1, 1, 1)
    ax.plot(x, costProgression)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('J(Theta)')
    fig.title('Convergence rate for learning rate alpha = {0}'.format(learningRate))
    return 


def plot_convergences(costProgressions, learningRates):
    """ Constructs a plot that compares convergence rates to respective learning rates. """
    
    learningRates = [str(alpha) for alpha in learningRates] # convert to string for column names
    costProgressions.columns = learningRates
    axes = costProgressions.plot()
    axes.legend(learningRates)
    return 


def calculate_normal_equation(data):
    """ Use the analytical method to calculate the optimal parameter vector. """
    
    # split data for legibility and insert the intercept
    features = data.iloc[:, :-1]
    features.insert(0, 'intercept', 1)
    response = data.iloc[:, -1]

    # compute the normal equation
    # this is kinda like using *= or += but for matrices instead
    theta = np.linalg.pinv(np.dot(features.T, features))
    theta = np.dot(theta, features.T)
    theta = np.dot(theta, response)
    return theta


def optional_main():
    """ Main function for optional exercises."""
    
    loc = r"C:\Users\ashamlian\Downloads\machine-learning-ex1\ex1\ex1data2.txt"
    housingData = load_data(loc, [0,1], [2])

    normedHousingData = normalize_features(housingData)

    # get a range of alphas and find the cost evolutions for them 
    testedLearningRates = [1, 0.3, .1, .03, .01, .003, .001]
    associatedCosts = []

    for alpha in testedLearningRates:

        _, costProgression = execute_gradient_descent(normedHousingData, learningRate=alpha, maxIters=100)
        associatedCosts.append(costProgression)

    associatedCosts = pd.concat(associatedCosts, axis=1) # now we have one nice pandas dataframe 
    plot_convergences(associatedCosts, testedLearningRates) # notice the resulting scale of the plot, still HUGE

    # we are going to predict the price of a 1650 sq. ft, 3 bedroom house
    # let's take the theta of the best of the learning rates: 1.0
    # but don't forget that we have to normalize the prediction for this model
    newHouse = [1650, 3]
    normedNewHouse = [-0.4412732002552997, 0.22367546144324044] # i did this manually, using the training mean() and std()
    fittedTheta, _ = execute_gradient_descent(normedHousingData, learningRate=1) # maxiters defaults to 2000
    print('For a 1650 square foot, 3 bedroom house, the predicted price is: {0}'.format(predict(normedNewHouse, fittedTheta)))

    # and compare it to that of the normal equation
    fittedTheta = calculate_normal_equation(housingData)
    print('For a 1650 square foot, 3 bedroom house, the predicted price is: {0}'.format(predict(newHouse, fittedTheta)))

    return 

if __name__ == '__main__':
    
    required_main()
    optional_main()