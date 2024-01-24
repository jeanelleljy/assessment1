# TASK 1
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np

data = 'jupiter.db'  # store the full dataset as "data"

class Moons:
    def __init__(self, data):
        self.conn = sqlite3.connect(data)
        self.data = pd.read_sql_query("SELECT * FROM moons", self.conn)
        
    def summary_statistics(self):
        return self.data.describe()  # returns summary stats like mean, median, sd, etc.
    
    def correlations(self):
        return self.data.corr()  # returns the correlations between variables in the dataset

    def plots(self, xcol, ycol):
        self.data.plot(kind='scatter', x=xcol, y=ycol)
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f'{xcol} vs {ycol}')
        plt.show()
        
    # extract data for 1 moon
    def get_moon_data(self, moon_name):
        return self.data[self.data['moon'] == moon_name]

    def __del__(self):
        self.conn.close()

# TASK 2

    def add_columns(self):  
        self.data['T2'] = (self.data['period_days']*60*60*24)**2  # Add new column T2 (converted to seconds)
        self.data['a3'] = (self.data['distance_km']*1000)**3  # Add new column a3 (converted to metres)
        
    def train_test_split(self):
        ## Create a testing and a training dataset
        # Prepare the input data, separating into training and testing sets
        X = self.data[['a3']]
        Y = self.data['T2']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=42)
        
    def linear_regression_model(self):
        self.model = linear_model.LinearRegression(fit_intercept=True)  # Create an instance of the model
        self.model.fit(self.X_train, self.Y_train)  # Train the model using the training set

    def estimate_jupiter_mass(self, G=6.67e-11):
        # Kepler's law: T^2 = (4 * pi^2 * a^3) / (G * M)
        # M = (4 * pi^2 * a^3) / (G * T^2)
        # coef = (4 * pi^2) / (G * M)
        # Hence M = (4 * pi^2) / (G * coef)        
        coef = self.model.coef_[0]
        jupiter_mass = (4 * np.pi**2) / (G * coef)
        return jupiter_mass
    
    def model_score(self):
        return self.model.score(self.X_test, self.Y_test)  # Returns the R2 score for this model
