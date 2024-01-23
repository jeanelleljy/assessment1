import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

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
