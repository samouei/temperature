# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Shirin Amouei
# Collaborators: None
# Time: 10:00
# Late Days Used: 1


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2008)
TESTING_INTERVAL = range(2008, 2017)

"""
Begin helper code
"""
class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

##########################
#    End helper code     #
##########################

def lin_regression(x, y):
    """
    Generate a linear regression model for the set of data points.

    Args:
        x: a list of length N, representing the x-coordinates of
            the N sample points
        y: a list of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                which are both floats.
    """
    
    # Average x and y values
    avg_x = sum(x) / len(x)
    avg_y = sum(y) / len(y)
    
    # Initialize numerator and denominator for calculating m
    num = 0
    denom = 0
    
    # Add up values
    for i in range(len(x)):
        num += (x[i] - avg_x) * (y[i] - avg_y)
        denom += (x[i] - avg_x)**2
    
    # Calculate slope and y-intercept
    m = num / denom
    b = avg_y - (m * avg_x)
    
    return (m, b)
    
def get_total_squared_error(x, y, m, b):
    '''
    Calculate the squared error of the regression model given the set of data points.

    Args:
        x: a list of length N, representing the x-coordinates of
            the N sample points
        y: a list of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        the total squared error of our regression
    '''
    
    # Calculate y values
    predicted_y_list = []
    for i in range(len(x)):
        predicted_y_list.append(m * x[i] + b)
    
    # Convert lists to arrays    
    predicted_y_array = np.array(predicted_y_list)
    observed_y_array = np.array(y)
    
    # Calculate squared error
    error = ((predicted_y_array - observed_y_array)**2).sum()
    
    return error
      
def make_models(x, y, degs):
    """
    Generate a polynomial regression model for each degree in degs, given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degs: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    
    # Create a list for storing array of coefficients
    model_list = []
    
    # Finf coefficients for each degree
    for degree in degs:
        model_list.append(np.polyfit(x, y, degree))
        
    return model_list

#print(make_models(np.array([1961, 1962, 1963]), np.array([-4.4, -5.5, -6.6]), [1, 2]))

def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value, and plot the data
    along with the best fit curve. For linear regression models (i.e. models with
    degree 1), you should also compute the SE/slope.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (i.e. the model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-squared of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope).

    R-squared and SE/slope should be rounded to 3 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    
    # For each model calculate predicted y and R^2
    for model in models:
        predicted_y = np.polyval(model, x) 
        R_squared = round(r2_score(y, predicted_y), 3)
        
        # Get SE for degree 1 models
        if len(model) == 2:
            SE = round(se_over_slope(x, y, predicted_y, model), 3)
        
        # Plotting ...
        plt.figure()
        if len(model) == 2:    
            plt.title("R^2 = " + str(R_squared) + ", Degree = " + str(len(model) - 1) + ", SE Over Slope = " + str(SE))
        else:
            plt.title("R^2 = " + str(R_squared) + ", Degree = " + str(len(model) - 1))
        
        # Plot curve
        plt.plot(x, predicted_y, 'r') 
        plt.xlabel("Years")
        plt.ylabel("Temperature (C)")
        
        # Plot data
        plt.plot(x, y, 'bo') 
        
        # Show plot
        plt.show()
        
def generate_cities_averages(temp, multi_cities, years):
    """
    For each year in the given range of years, computes the average of the
    annual temperatures in the given cities.

    Args:
        temp: instance of Dataset
        multi_cities: (list of str) the names of cities to include in the average
            annual temperature calculation
        years: (list of int) the range of years of the annual averaged temperatures

    Returns:
        a numpy 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """

    average_annual_temps = []
    
    # For each year, get average annual temperature for all citites
    for year in years:
        multi_cities_sum = 0
        for city in multi_cities:
            total_year_temp = temp.get_yearly_temp(city, year)
            average_year_temp = total_year_temp.sum()/ len(total_year_temp)
            multi_cities_sum += average_year_temp
        
        average_annual_temps.append(multi_cities_sum / len(multi_cities))
    
    return np.array(average_annual_temps) 
    
def find_interval(x, y, length, use_positive_slope):
    """
    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        use_positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope and j-i = length.

        In the case of a tie, it returns the most recent interval. For example,
        if the intervals (2,5) and (8,11) both have the same slope, (8,11) should
        be returned.

        If such an interval does not exist, returns None
    """
    
    # Initialize best slope
    best_slope = 0
    
    # If no interval return None
    best_interval = None
    
    # Get slope for each interval of lenght "lenght"
    for i in range(len(x) - length + 1):
        j = i + length
        slope, b = lin_regression(x[i:j], y[i:j])
        
        # Compare slope with previous slope
        # Return interval
        if use_positive_slope:
            if slope >= best_slope:
                best_slope = slope
                best_interval = (i, j)    
        else:
            if slope <= best_slope:
                best_slope = slope
                best_interval = (i, j)
                
    return best_interval
                 
def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    
    total = 0
    
    # Calculate RMSE numerator
    for i in range(len(y)):
        total += (y[i] - estimated[i])**2
        
    return (total / len(y))**0.5
        
def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the 
    test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 3 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    
    # Calculate predicted y and RMSE for each model
    for model in models:
        predicted_y = np.polyval(model, x) 
        RMSE = round(rmse(y, predicted_y), 3)
    
        # Plotting ...
        plt.figure()
        plt.title("RMSE = " + str(RMSE) + ", Degree = " + str(len(model) - 1))
        
        # Plot curve
        plt.plot(x, predicted_y, 'r') 
        plt.xlabel("Years")
        plt.ylabel("Temperature (C)")
        
        # Plot data
        plt.plot(x, y, 'bo') 
        
        # Show plot
        plt.show()
    

if __name__ == '__main__':

    pass
    
    ###############
    # Problem 4A
    
#    data = Dataset("data.csv")
#    x = range(1961, 2016)
#    y_vals = []
#    for year in range(1961, 2016):
#        y = data.get_daily_temp('PORTLAND', 12, 25, year)
#        y_vals.append(y)
#    
#    model = make_models(np.array(x), np.array(y_vals), [1]) 
#    evaluate_models_on_training(np.array(x), np.array(y_vals), model)

    ###############
    # Problem 4B
    
#    data = Dataset("data.csv")
#    x = range(1961, 2016)
#    y_vals = generate_cities_averages(data, ['PORTLAND'], x)
#
#    model = make_models(np.array(x), y_vals, [1]) 
#    evaluate_models_on_training(np.array(x), y_vals, model)

    ###############
    # Problem 5B
    
#    data = Dataset("data.csv")
#    x = range(1961, 2016)
#    y_vals = generate_cities_averages(data, ['SAN FRANCISCO'], x)
#    i, j = find_interval(x, y_vals, 30, True)
#    
#    model = make_models(np.array(x[i:j]), y_vals[i:j], [1]) 
#    evaluate_models_on_training(np.array(x[i:j]), y_vals[i:j], model)
#    
#    i, j = find_interval(x, y_vals, 30, True)
#    print(x[i:j], y_vals[i:j])
#    print(lin_regression(x[i:j], y_vals[i:j]))
    
    ###############
    # Problem 5C
    
#    data = Dataset("data.csv")
#    x = range(1961, 2016)
#    y_vals = generate_cities_averages(data, ['SAN FRANCISCO'], x)
#    i, j = find_interval(x, y_vals, 20, False)
#    
#    model = make_models(np.array(x[i:j]), y_vals[i:j], [1]) 
#    evaluate_models_on_training(np.array(x[i:j]), y_vals[i:j], model)
#    
#    i, j = find_interval(x, y_vals, 20, False)
#    print(x[i:j], y_vals[i:j])
#    print(lin_regression(x[i:j], y_vals[i:j]))

    ###############
    # Problem 6B
    
#    data = Dataset("data.csv")
#    TRAINING_INTERVAL = range(1961, 2008)
#    y_vals = generate_cities_averages(data, CITIES, TRAINING_INTERVAL)
#    
#    model = make_models(np.array(TRAINING_INTERVAL), y_vals, [2, 10]) 
#    evaluate_models_on_training(np.array(TRAINING_INTERVAL), y_vals, model)
#    
#    TESTING_INTERVAL = range(2008, 2017)
#    y_vals = generate_cities_averages(data, CITIES, TESTING_INTERVAL)
#    evaluate_models_on_testing(np.array(TESTING_INTERVAL), y_vals, model)
    
    
    
    
    
