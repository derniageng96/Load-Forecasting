"""
    Data Imputation
    We have to fill the missing values with another value to complete our dataset. There are several methods to do that. 

    In this notebook we present Random Forest, Interpolation, and Average. 
    We only need to select one of them

    ## Random Forest
    """
# import plotly.graph_objects as go
# import plotly.io as pio
# import chart_studio
# import plotly.express as px
# chart_studio.tools.set_credentials_file(username='DerniAgeng', api_key='ZMAguW1HDlV8v7EYvqVJ')
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as md

class Imputation:
    """
    Imputation contains 3 functions
    
    Attributes:
        interpolation (dataframe) : fill the missing values with average value of previous and next values
        randomforest (dataframe) : fill the missing values with random forest algorithm
        avg (dataframe) : fill the missing values with average value
    
    """
    def interpolation(self, id1,id2,id3,id4,id5):
        """
        fill the missing values with average value of previous and next values
        
        Parameters:
            id1 (dataframe) : building id 1 dataset
            mt2
        
        Returns:
            combine (dataframe) : processed dataset
        """

        mt1= mt1.interpolate()
        mt2= mt2.interpolate()
        mt3= mt3.interpolate()
        mt4= mt4.interpolate()
        mt5= mt5.interpolate()
        
        return mt1,mt2,mt3,mt4,mt5

    def randomforest(self,mt1,mt2,mt3,mt4,mt5):
            
        #MT1
        mt1withpsum = mt1[pd.isna(mt1['p_sum']) == False]
        mt1withoutpsum = mt1[pd.isna(mt1['p_sum'])]
        #MT2
        mt2withpsum = mt2[pd.isna(mt2['p_sum']) == False]
        mt2withoutpsum = mt2[pd.isna(mt2['p_sum'])]
        #MT3
        mt3withpsum = mt3[pd.isna(mt3['p_sum']) == False]
        mt3withoutpsum = mt3[pd.isna(mt3['p_sum'])]


        #MT4
        mt4withpsum = mt4[pd.isna(mt4['p_sum']) == False]
        mt4withoutpsum = mt4[pd.isna(mt4['p_sum'])]
        #MT5
        mt5withpsum = mt5[pd.isna(mt5['p_sum']) == False]
        mt5withoutpsum = mt5[pd.isna(mt5['p_sum'])]

        variables = ['meterType','slaveAddr', 'blockId','wire','freq','ae_tot']


        rf = RandomForestRegressor()
        rf.fit(mt1withpsum[variables], mt1withpsum['p_sum'])
        rf.fit(mt2withpsum[variables], mt2withpsum['p_sum'])
        rf.fit(mt3withpsum[variables], mt3withpsum['p_sum'])
        rf.fit(mt4withpsum[variables], mt4withpsum['p_sum'])
        rf.fit(mt5withpsum[variables], mt5withpsum['p_sum'])

        generated_psum1 = rf.predict(X = mt1withoutpsum[variables])
        generated_psum2 = rf.predict(X = mt2withoutpsum[variables])
        generated_psum3 = rf.predict(X = mt3withoutpsum[variables])
        generated_psum4 = rf.predict(X = mt4withoutpsum[variables])
        generated_psum5 = rf.predict(X = mt5withoutpsum[variables])

        len(generated_psum2)

        mt2withoutpsum['p_sum'] = generated_psum2.astype(int)

        mt2 = mt2withpsum.append(mt2withoutpsum)

    """Average"""

    def avg(self,mt1,mt2,mt3,mt4,mt5):

        mt1['p_sum'] = mt1['p_sum'].fillna(mt1['p_sum'].mean())
        mt2['p_sum'] = mt2['p_sum'].fillna(mt2['p_sum'].mean())
        mt3['p_sum'] = mt3['p_sum'].fillna(mt3['p_sum'].mean())
        mt4['p_sum'] = mt4['p_sum'].fillna(mt4['p_sum'].mean())
        mt5['p_sum'] = mt5['p_sum'].fillna(mt5['p_sum'].mean())

def main():
    mt1 = pd.read_csv('MT1_withnan_202007-202104.csv')
    mt2 = pd.read_csv('MT2_withnan_202007-202104.csv')
    mt3 = pd.read_csv('MT3_withnan_202007-202104.csv')
    mt4 = pd.read_csv('MT4_withnan_202007-202104.csv')
    mt5 = pd.read_csv('MT5_withnan_202007-202104.csv')

    imp = Imputation()
    inter = imp.interpolation(mt1,mt2,mt3,mt4,mt5)
    avrg = imp.avg(mt1,mt2,mt3,mt4,mt5)

if __name__ == '__main__':
    main()    
        