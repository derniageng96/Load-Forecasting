"""# Import Library
## Use Plotly is optional, Plotly chose because we can filter the data to be shown
### You can use Matplotlib as well
"""

import plotly.graph_objects as go
import plotly.io as pio
import chart_studio
import plotly.express as px
chart_studio.tools.set_credentials_file(username='DerniAgeng', api_key='ZMAguW1HDlV8v7EYvqVJ')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn 
import matplotlib.dates as md

"""Data Exploratory
"""
class Preprocessing:
    """
    Preprocessing contains 4 functions
    
    Attributes:
        tr (dataframe) : data preprocessing for training building
        chang (dataframe) : data preprocessing for changyuan building
        mt (dataframe) : data preprocessing for maintenance building
        analysis (dataframe) : data analysis for selected dataframe
    
    """

    def tr(self,data):
        """
        preprocessing for training building data
        
        Parameters:
            data (dataframe) : training building dataset
        
        Returns:
            combine (dataframe) : processed dataset
        """

        missing_value = data.isna().mean() * 100
        missing_value.plot(kind='bar', figsize=(15,10))
        plt.ylabel('Missing Percentage', size=15)
        plt.xlabel('Features', size=15)
        plt.show()

        data = data.loc[:, data.isna().mean() < .30]

        tr1 = data[data['equipmentId'].isin(['001003304511002487000001'])]
        tr2 = data[data['equipmentId'].isin(['001003304511002487000002'])]
        tr3 = data[data['equipmentId'].isin(['001003304511002487000003'])]
        tr4 = data[data['equipmentId'].isin(['001003304511002487000004'])]
        tr5 = data[data['equipmentId'].isin(['001003304511002487000005'])]
        tr1.head()

        vars_num_anom = [var for var in tr1.columns if data[var].dtypes != 'O']
        tr2[vars_num_anom].head()

        tr1.columns

        tr1[vars_num_anom] = tr1[vars_num_anom].replace(
            {2147483647:np.NaN})
        tr2[vars_num_anom] = tr2[vars_num_anom].replace(
            {2147483647:np.NaN})
        tr3[vars_num_anom] = tr3[vars_num_anom].replace(
            {2147483647:np.NaN})
        tr4[vars_num_anom] = tr4[vars_num_anom].replace(
            {2147483647:np.NaN})
        tr5[vars_num_anom] = tr5[vars_num_anom].replace(
            {2147483647:np.NaN})

        tr1.to_csv('TR1_withnan_202007-202104.csv')
        tr2.to_csv('TR2_withnan_202007-202104.csv')
        tr3.to_csv('TR3_withnan_202007-202104.csv')
        tr4.to_csv('TR4_withnan_202007-202104.csv')
        tr5.to_csv('TR5_withnan_202007-202104.csv')
        
        """# Data Imputation
        
        ## Interpolation
        """
        tr1= tr1.interpolate()
        tr2= tr2.interpolate()
        tr3= tr3.interpolate()
        tr4= tr4.interpolate()
        tr5= tr5.interpolate()

        """# Resample each EQID in Training Building"""

        tr1['lastReportTime'] = pd.to_datetime(tr1['lastReportTime'], errors='coerce')
        tr1.set_index('lastReportTime', inplace=True)
        tr1 = tr1.resample('h').mean()
        tr1 = tr1.reset_index()
        tr2['lastReportTime'] = pd.to_datetime(tr2['lastReportTime'], errors='coerce')
        tr2.set_index('lastReportTime', inplace=True)
        tr2 = tr2.resample('h').mean()
        tr2 = tr2.reset_index()
        tr3['lastReportTime'] = pd.to_datetime(tr3['lastReportTime'], errors='coerce')
        tr3.set_index('lastReportTime', inplace=True)
        tr3 = tr3.resample('h').mean()
        tr3 = tr3.reset_index()
        tr4['lastReportTime'] = pd.to_datetime(tr4['lastReportTime'], errors='coerce')
        tr4.set_index('lastReportTime', inplace=True)
        tr4 = tr4.resample('h').mean()
        tr4 = tr4.reset_index()
        tr5['lastReportTime'] = pd.to_datetime(tr5['lastReportTime'], errors='coerce')
        tr5.set_index('lastReportTime', inplace=True)
        tr5 = tr5.resample('h').mean()
        tr5 = tr5.reset_index()

        # fill the missing values again if there is still 'NaN' value
        tr1= tr1.interpolate()
        tr2= tr2.interpolate()
        tr3= tr3.interpolate()
        tr4= tr4.interpolate()
        tr5= tr5.interpolate()

        """Since there is big gap in the first part of dataset we only chose this range of dataset"""

        tr1 = tr1[900:4080]
        tr2 = tr2[900:4080]
        tr3 = tr3[900:4080]
        tr4 = tr4[900:4080]
        tr5 = tr5[900:4080]

        """# Feature Engineering

        ## Create Addition Feature
        """

        tr1['lastReportTime'] = pd.to_datetime(tr1['lastReportTime'], errors='coerce')
        tr1 = tr1.assign(session=pd.cut(tr1['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        tr2['lastReportTime'] = pd.to_datetime(tr2['lastReportTime'], errors='coerce')
        tr2 = tr2.assign(session=pd.cut(tr2['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        tr3['lastReportTime'] = pd.to_datetime(tr3['lastReportTime'], errors='coerce')
        tr3 = tr3.assign(session=pd.cut(tr3['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        tr4['lastReportTime'] = pd.to_datetime(tr4['lastReportTime'], errors='coerce')
        tr4 = tr4.assign(session=pd.cut(tr4['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        tr5['lastReportTime'] = pd.to_datetime(tr5['lastReportTime'], errors='coerce')
        tr5 = tr5.assign(session=pd.cut(tr5['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))

        tr1.head()

        tr1['session'] = tr1['session'].cat.add_categories('Midnight')
        tr1['session'] = tr1['session'].fillna('Midnight')
        tr2['session'] = tr2['session'].cat.add_categories('Midnight')
        tr2['session'] = tr2['session'].fillna('Midnight')
        tr3['session'] = tr3['session'].cat.add_categories('Midnight')
        tr3['session'] = tr3['session'].fillna('Midnight')
        tr4['session'] = tr4['session'].cat.add_categories('Midnight')
        tr4['session'] = tr4['session'].fillna('Midnight')
        tr5['session'] = tr5['session'].cat.add_categories('Midnight')
        tr5['session'] = tr5['session'].fillna('Midnight')

        tr1['session'] = tr1['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        tr2['session'] = tr2['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        tr3['session'] = tr3['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        tr4['session'] = tr4['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        tr5['session'] = tr5['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})

        tr1['weekend'] = np.where((tr1['lastReportTime']).dt.dayofweek < 5,0,1)
        tr2['weekend'] = np.where((tr2['lastReportTime']).dt.dayofweek < 5,0,1)
        tr3['weekend'] = np.where((tr3['lastReportTime']).dt.dayofweek < 5,0,1)
        tr4['weekend'] = np.where((tr4['lastReportTime']).dt.dayofweek < 5,0,1)
        tr5['weekend'] = np.where((tr5['lastReportTime']).dt.dayofweek < 5,0,1)

        """## Merge with Temperature"""

        d_two = pd.read_csv('Taipei_temperature_202007-202104.csv',sep=',',engine='python')

        d_two['lastReportTime'] = pd.to_datetime(d_two['lastReportTime'], errors='coerce')
        d_two = d_two[['Temperature', 'lastReportTime']]

        d_two.set_index('lastReportTime', inplace = True)
        d_two = d_two.resample('h').mean()
        d_two = d_two.reset_index()
        d_two.head()

        trid1 = pd.merge(tr1, d_two, left_on='lastReportTime', right_on='lastReportTime')
        trid2 = pd.merge(tr2, d_two, left_on='lastReportTime', right_on='lastReportTime')
        trid3 = pd.merge(tr3, d_two, left_on='lastReportTime', right_on='lastReportTime')
        trid4 = pd.merge(tr4, d_two, left_on='lastReportTime', right_on='lastReportTime')
        trid5 = pd.merge(tr5, d_two, left_on='lastReportTime', right_on='lastReportTime')

        trid2.head()

        trid1['equipmentId'] = '001003304511002487000001'
        trid2['equipmentId'] = '001003304511002487000002'
        trid3['equipmentId'] = '001003304511002487000003'
        trid4['equipmentId'] = '001003304511002487000004'
        trid5['equipmentId'] = '001003304511002487000005'

        """# Combine every EQID"""

        combine = pd.concat([trid1,trid2,trid3,trid4,trid5])
        combine

        combine['lastReportTime'] = pd.to_datetime(combine['lastReportTime'], errors='coerce')
        combine.set_index('lastReportTime', inplace=True)
        combine = combine.resample('h').mean()
        combine = combine.reset_index()
        combine.head()

        combine.to_csv('trainingbuilding_interpolation_202007-202104')
        combine

    def chang(self, data):
        """
        preprocessing for changyuan building data
        
        Parameters:
            data (dataframe) : changyuan dataset
        
        Returns:
            combine (dataframe) : processed dataset
        """
        
        """## Select EQID from Changyuan building"""

        changyuan = data[data['equipmentId'].isin(['001003f4e11ed5c10c000002'])]
        changyuan.head()

        """## Separate Numerical Column"""

        vars_num_anom = [var for var in changyuan.columns if data[var].dtypes != 'O']
        changyuan[vars_num_anom].head()

        """## Check Missing Percentage"""

        missing_value = changyuan.isna().mean() * 100
        missing_value.plot(kind='bar', figsize=(15,10))
        plt.ylabel('Missing Percentage', size=15)
        plt.xlabel('Features', size=15)
        plt.show()

        """## Replace the wrong value to 'NaN'"""

        changyuan[vars_num_anom] = changyuan[vars_num_anom].replace(
            {2147483647:np.NaN})

        """## Drop the column that has missing percentage greater than 30%"""

        changyuan = changyuan.loc[:, changyuan.isna().mean() < .30]
        changyuan.columns

        """Check Missing percentage of each ID"""

        c1 = changyuan[changyuan['blockId'].isin(['1'])]
        c2 = changyuan[changyuan['blockId'].isin(['2'])]
        c3 = changyuan[changyuan['blockId'].isin(['3'])]
        c4 = changyuan[changyuan['blockId'].isin(['4'])]
        c5 = changyuan[changyuan['blockId'].isin(['5'])]
        c6 = changyuan[changyuan['blockId'].isin(['6'])]
        c7 = changyuan[changyuan['blockId'].isin(['7'])]
        c8 = changyuan[changyuan['blockId'].isin(['8'])]

        c1.to_csv('C1_withnan_202007-202104.csv')
        c2.to_csv('C2_withnan_202007-202104.csv')
        c3.to_csv('C3_withnan_202007-202104.csv')    
        c4.to_csv('C4_withnan_202007-202104.csv')
        c5.to_csv('C5_withnan_202007-202104.csv')
        c6.to_csv('C6_withnan_202007-202104.csv')
        c7.to_csv('C7_withnan_202007-202104.csv')
        c8.to_csv('C8_withnan_202007-202104.csv')
    
        c1 = c1.interpolate()
        c2 = c2.interpolate()
        c3 = c3.interpolate()
        c4 = c4.interpolate()
        c5 = c5.interpolate()
        c6 = c6.interpolate()
        c7 = c7.interpolate()
        c8 = c8.interpolate()

        """# Resample each block"""

        c1['lastReportTime'] = pd.to_datetime(c1['lastReportTime'], errors='coerce')
        c1.set_index('lastReportTime', inplace=True)
        c1 = c1.resample('h').mean()
        c1 = c1.reset_index()
        c2['lastReportTime'] = pd.to_datetime(c2['lastReportTime'], errors='coerce')
        c2.set_index('lastReportTime', inplace=True)
        c2 = c2.resample('h').mean()
        c2 = c2.reset_index()
        c3['lastReportTime'] = pd.to_datetime(c3['lastReportTime'], errors='coerce')
        c3.set_index('lastReportTime', inplace=True)
        c3 = c3.resample('h').mean()
        c3 = c3.reset_index()
        c4['lastReportTime'] = pd.to_datetime(c4['lastReportTime'], errors='coerce')
        c4.set_index('lastReportTime', inplace=True)
        c4 = c4.resample('h').mean()
        c4 = c4.reset_index()
        c5['lastReportTime'] = pd.to_datetime(c5['lastReportTime'], errors='coerce')
        c5.set_index('lastReportTime', inplace=True)
        c5 = c5.resample('h').mean()
        c5 = c5.reset_index()
        c6['lastReportTime'] = pd.to_datetime(c6['lastReportTime'], errors='coerce')
        c6.set_index('lastReportTime', inplace=True)
        c6 = c6.resample('h').mean()
        c6 = c6.reset_index()
        c7['lastReportTime'] = pd.to_datetime(c7['lastReportTime'], errors='coerce')
        c7.set_index('lastReportTime', inplace=True)
        c7 = c7.resample('h').mean()
        c7 = c7.reset_index()
        c8['lastReportTime'] = pd.to_datetime(c8['lastReportTime'], errors='coerce')
        c8.set_index('lastReportTime', inplace=True)
        c8 = c8.resample('h').mean()
        c8 = c8.reset_index()

        c1.to_csv('C1_202007-202104.csv')
        c2.to_csv('C2_202007-202104.csv')
        c3.to_csv('C3_202007-202104.csv')
        c4.to_csv('C4_202007-202104.csv')
        c5.to_csv('C5_202007-202104.csv')
        c6.to_csv('C6_202007-202104.csv')
        c7.to_csv('C7_202007-202104.csv')
        c8.to_csv('C8_202007-202104.csv')

        """# Feature Engineering
        ## Create Additional Feature
        """

        c1['lastReportTime'] = pd.to_datetime(c1['lastReportTime'], errors='coerce')
        c1 = c1.assign(session=pd.cut(c1['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c2['lastReportTime'] = pd.to_datetime(c2['lastReportTime'], errors='coerce')
        c2 = c2.assign(session=pd.cut(c2['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c3['lastReportTime'] = pd.to_datetime(c3['lastReportTime'], errors='coerce')
        c3 = c3.assign(session=pd.cut(c3['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c4['lastReportTime'] = pd.to_datetime(c4['lastReportTime'], errors='coerce')
        c4 = c4.assign(session=pd.cut(c4['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c5['lastReportTime'] = pd.to_datetime(c5['lastReportTime'], errors='coerce')
        c5 = c5.assign(session=pd.cut(c5['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c6['lastReportTime'] = pd.to_datetime(c6['lastReportTime'], errors='coerce')
        c6 = c6.assign(session=pd.cut(c6['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c7['lastReportTime'] = pd.to_datetime(c7['lastReportTime'], errors='coerce')
        c7 = c7.assign(session=pd.cut(c7['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        c8['lastReportTime'] = pd.to_datetime(c8['lastReportTime'], errors='coerce')
        c8 = c8.assign(session=pd.cut(c8['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))

        c1['session'] = c1['session'].cat.add_categories('Midnight')
        c1['session'] = c1['session'].fillna('Midnight')
        c2['session'] = c2['session'].cat.add_categories('Midnight')
        c2['session'] = c2['session'].fillna('Midnight')
        c3['session'] = c3['session'].cat.add_categories('Midnight')
        c3['session'] = c3['session'].fillna('Midnight')
        c4['session'] = c4['session'].cat.add_categories('Midnight')
        c4['session'] = c4['session'].fillna('Midnight')
        c5['session'] = c5['session'].cat.add_categories('Midnight')
        c5['session'] = c5['session'].fillna('Midnight')
        c6['session'] = c6['session'].cat.add_categories('Midnight')
        c6['session'] = c6['session'].fillna('Midnight')
        c7['session'] = c7['session'].cat.add_categories('Midnight')
        c7['session'] = c7['session'].fillna('Midnight')
        c8['session'] = c8['session'].cat.add_categories('Midnight')
        c8['session'] = c8['session'].fillna('Midnight')

        c1['session'] = c1['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c2['session'] = c2['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c3['session'] = c3['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c4['session'] = c4['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c5['session'] = c5['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c6['session'] = c6['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c7['session'] = c7['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        c8['session'] = c8['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})

        c1['weekend'] = np.where((c1['lastReportTime']).dt.dayofweek < 5,0,1)
        c2['weekend'] = np.where((c2['lastReportTime']).dt.dayofweek < 5,0,1)
        c3['weekend'] = np.where((c3['lastReportTime']).dt.dayofweek < 5,0,1)
        c4['weekend'] = np.where((c4['lastReportTime']).dt.dayofweek < 5,0,1)
        c5['weekend'] = np.where((c5['lastReportTime']).dt.dayofweek < 5,0,1)
        c6['weekend'] = np.where((c6['lastReportTime']).dt.dayofweek < 5,0,1)
        c7['weekend'] = np.where((c7['lastReportTime']).dt.dayofweek < 5,0,1)
        c8['weekend'] = np.where((c8['lastReportTime']).dt.dayofweek < 5,0,1)

        """# Merge with Temperature Dataset"""

        d_two = pd.read_csv('Taipei_temperature_202007-202104.csv',sep=',',engine='python')

        d_two['lastReportTime'] = pd.to_datetime(d_two['lastReportTime'], errors='coerce')
        d_two = d_two[['Temperature', 'lastReportTime']]

        cid1 = pd.merge(c1, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid2 = pd.merge(c2, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid3 = pd.merge(c3, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid4 = pd.merge(c4, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid5 = pd.merge(c5, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid6 = pd.merge(c6, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid7 = pd.merge(c7, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid8 = pd.merge(c8, d_two, left_on='lastReportTime', right_on='lastReportTime')
        cid1.head()

        """# Combine every ID"""

        combine = pd.concat([cid1,cid2,cid3,cid4,cid5,cid6,cid7,cid8])
        combine

        combine['lastReportTime'] = pd.to_datetime(combine['lastReportTime'], errors='coerce')
        combine.set_index('lastReportTime', inplace=True)
        combine = combine.resample('h').mean()
        combine = combine.reset_index()
        
        combine = combine.interpolate()
        combine.to_csv('changyuanbuilding(202007-202104)_interpolation.csv')
        
        return combine
        
    def mt(self, data):
        """
        preprocessing for maintenance building data
        
        Parameters:
            data (dataframe) : maintenance dataset
        
        Returns:
            combine (dataframe) : processed dataset
        """


        """Check missing percentage"""

        missing_value = data.isna().mean() * 100
        missing_value.plot(kind='bar', figsize=(15,10))
        plt.ylabel('Missing Percentage', size=15)
        plt.xlabel('Features', size=15)
        plt.show()

        """Drop Column that has missing percentage greater than 35%"""

        data = data.loc[:, data.isna().mean() < .35]
        data.columns
        data['ae_tot'] = pd.to_numeric(data['ae_tot'])

        """Separate each ID to different dataframe"""

        mt1 = data[data['equipmentId'].isin(['001003f4e11edd022d000001'])]
        mt2 = data[data['equipmentId'].isin(['001003f4e11edd022d000002'])]
        mt3 = data[data['equipmentId'].isin(['001003f4e11edd022d000003'])]
        mt4 = data[data['equipmentId'].isin(['001003f4e11edd022d000004'])]
        mt5 = data[data['equipmentId'].isin(['001003f4e11edd022d000005'])]
        mt1.head()

        mt5['blockId'].unique()


        """Convert 'LastReportTime' column as datetime"""

        mt1['lastReportTime'] = pd.to_datetime(mt1['lastReportTime'], errors='coerce')
        mt1.set_index('lastReportTime', inplace=True)
        # mt1 = mt1.resample('h').mean()
        mt1 = mt1.reset_index()
        mt2['lastReportTime'] = pd.to_datetime(mt2['lastReportTime'], errors='coerce')
        mt2.set_index('lastReportTime', inplace=True)
        # mt2 = mt2.resample('h').mean()
        mt2 = mt2.reset_index()
        mt3['lastReportTime'] = pd.to_datetime(mt3['lastReportTime'], errors='coerce')
        mt3.set_index('lastReportTime', inplace=True)
        # mt3 = mt3.resample('h').mean()
        mt3 = mt3.reset_index()
        mt4['lastReportTime'] = pd.to_datetime(mt4['lastReportTime'], errors='coerce')
        mt4.set_index('lastReportTime', inplace=True)
        # mt4 = mt4.resample('h').mean()
        mt4 = mt4.reset_index()
        mt5['lastReportTime'] = pd.to_datetime(mt5['lastReportTime'], errors='coerce')
        mt5.set_index('lastReportTime', inplace=True)
        # mt5 = mt5.resample('h').mean()
        mt5 = mt5.reset_index()

        """Handle the Wrong Value
        We observed the load data in particular time, and found that overflow value sent to the database(2^31)

        Reference : [Overflow](https://microchipdeveloper.com/dsp0201:overflow-and-saturation)
        """
        
        """Separate numerical column on dataset"""

        vars_num_anom = [var for var in mt1.columns if data[var].dtypes != 'O']
        mt2[vars_num_anom].head()

        mt1.columns

        """Replace the wrong value with NaN"""

        mt1[vars_num_anom] = mt1[vars_num_anom].replace(
            {2147483647:np.NaN})
        mt2[vars_num_anom] = mt2[vars_num_anom].replace(
            {2147483647:np.NaN})
        mt3[vars_num_anom] = mt3[vars_num_anom].replace(
            {2147483647:np.NaN})
        mt4[vars_num_anom] = mt4[vars_num_anom].replace(
            {2147483647:np.NaN})
        mt5[vars_num_anom] = mt5[vars_num_anom].replace(
            {2147483647:np.NaN})

        mt5.isna().mean()*100

        mt1.to_csv('MT1_withnan_202007-202104.csv')
        mt2.to_csv('MT2_withnan_202007-202104.csv')
        mt3.to_csv('MT3_withnan_202007-202104.csv')
        mt4.to_csv('MT4_withnan_202007-202104.csv')
        mt5.to_csv('MT5_withnan_202007-202104.csv')
        
        """Interpolation"""
        mt1= mt1.interpolate()
        mt2= mt2.interpolate()
        mt3= mt3.interpolate()
        mt4= mt4.interpolate()
        mt5= mt5.interpolate()
        mt2.head()

        """Resample each EQID in Maintenance Building"""

        mt1['lastReportTime'] = pd.to_datetime(mt1['lastReportTime'], errors='coerce')
        mt1.set_index('lastReportTime', inplace=True)
        mt1 = mt1.resample('h').mean()
        mt1 = mt1.reset_index()
        mt2['lastReportTime'] = pd.to_datetime(mt2['lastReportTime'], errors='coerce')
        mt2.set_index('lastReportTime', inplace=True)
        mt2 = mt2.resample('h').mean()
        mt2 = mt2.reset_index()
        mt3['lastReportTime'] = pd.to_datetime(mt3['lastReportTime'], errors='coerce')
        mt3.set_index('lastReportTime', inplace=True)
        mt3 = mt3.resample('h').mean()
        mt3 = mt3.reset_index()
        mt4['lastReportTime'] = pd.to_datetime(mt4['lastReportTime'], errors='coerce')
        mt4.set_index('lastReportTime', inplace=True)
        mt4 = mt4.resample('h').mean()
        mt4 = mt4.reset_index()
        mt5['lastReportTime'] = pd.to_datetime(mt5['lastReportTime'], errors='coerce')
        mt5.set_index('lastReportTime', inplace=True)
        mt5 = mt5.resample('h').mean()
        mt5 = mt5.reset_index()

        mt1['equipmentId'] = '001003f4e11edd022d000001'
        mt2['equipmentId'] = '001003f4e11edd022d000002'
        mt3['equipmentId'] = '001003f4e11edd022d000003'
        mt4['equipmentId'] = '001003f4e11edd022d000004'
        mt5['equipmentId'] = '001003f4e11edd022d000005'

        mt1.head()

        mt1.to_csv('MT1(2020_07-2021_04)_interpolation.csv')
        mt2.to_csv('MT2(2020_07-2021_04)_interpolation.csv')
        mt3.to_csv('MT3(2020_07-2021_04)_interpolation.csv')
        mt4.to_csv('MT4(2020_07-2021_04)_interpolation.csv')
        mt5.to_csv('MT5(2020_07-2021_04)_interpolation.csv')

        """# Feature Engineering
        ## Create Additional Feature (Session Time, Weekend/Weekday indicator)
        """

        mt1['lastReportTime'] = pd.to_datetime(mt1['lastReportTime'], errors='coerce')
        mt1 = mt1.assign(session=pd.cut(mt1['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        mt2['lastReportTime'] = pd.to_datetime(mt2['lastReportTime'], errors='coerce')
        mt2 = mt2.assign(session=pd.cut(mt2['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        mt3['lastReportTime'] = pd.to_datetime(mt3['lastReportTime'], errors='coerce')
        mt3 = mt3.assign(session=pd.cut(mt3['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        mt4['lastReportTime'] = pd.to_datetime(mt4['lastReportTime'], errors='coerce')
        mt4 = mt4.assign(session=pd.cut(mt4['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
        mt5['lastReportTime'] = pd.to_datetime(mt5['lastReportTime'], errors='coerce')
        mt5 = mt5.assign(session=pd.cut(mt5['lastReportTime'].dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))

        mt1.head()

        mt1['session'] = mt1['session'].cat.add_categories('Midnight')
        mt1['session'] = mt1['session'].fillna('Midnight')
        mt2['session'] = mt2['session'].cat.add_categories('Midnight')
        mt2['session'] = mt2['session'].fillna('Midnight')
        mt3['session'] = mt3['session'].cat.add_categories('Midnight')
        mt3['session'] = mt3['session'].fillna('Midnight')
        mt4['session'] = mt4['session'].cat.add_categories('Midnight')
        mt4['session'] = mt4['session'].fillna('Midnight')
        mt5['session'] = mt5['session'].cat.add_categories('Midnight')
        mt5['session'] = mt5['session'].fillna('Midnight')

        mt1['session'] = mt1['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        mt2['session'] = mt2['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        mt3['session'] = mt3['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        mt4['session'] = mt4['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})
        mt5['session'] = mt5['session'].replace({'Night':0,'Midnight':1,'Morning':2,'Afternoon':3,'Evening':4})

        mt1['weekend'] = np.where((mt1['lastReportTime']).dt.dayofweek < 5,0,1)
        mt2['weekend'] = np.where((mt2['lastReportTime']).dt.dayofweek < 5,0,1)
        mt3['weekend'] = np.where((mt3['lastReportTime']).dt.dayofweek < 5,0,1)
        mt4['weekend'] = np.where((mt4['lastReportTime']).dt.dayofweek < 5,0,1)
        mt5['weekend'] = np.where((mt5['lastReportTime']).dt.dayofweek < 5,0,1)

        """Merge with Temperature"""

        d_two = pd.read_csv('Taipei_temperature_202007-202104.csv',sep=',',engine='python')

        d_two['lastReportTime'] = pd.to_datetime(d_two['lastReportTime'], errors='coerce')
        d_two = d_two[['Temperature', 'lastReportTime']]

        mtid1 = pd.merge(mt1, d_two, left_on='lastReportTime', right_on='lastReportTime')
        mtid2 = pd.merge(mt2, d_two, left_on='lastReportTime', right_on='lastReportTime')
        mtid3 = pd.merge(mt3, d_two, left_on='lastReportTime', right_on='lastReportTime')
        mtid4 = pd.merge(mt4, d_two, left_on='lastReportTime', right_on='lastReportTime')
        mtid5 = pd.merge(mt5, d_two, left_on='lastReportTime', right_on='lastReportTime')

        mtid1

        """Combine every EQID"""

        combine = pd.concat([mtid1,mtid2,mtid3,mtid4,mtid5])
        combine.head()

        combine['lastReportTime'] = pd.to_datetime(combine['lastReportTime'], errors='coerce')
        combine.set_index('lastReportTime', inplace=True)
        combine = combine.resample('h').mean()
        combine = combine.reset_index()
        combine.head()

        combine.to_csv('mtbuilding_cleanedwithinterpolation_202007-202104.csv')
        combine.head()

        return combine
    
    """
        Data Analysis
        Pattern on Dataset
    """
    def analysis(self, data):
        """
        Analysis load data
        
        Parameters:
            data (dataframe) : load data
        """
        # data = pd.read_csv('mtbuilding_cleanedwithinterpolation_202007-202104.csv')
        data['lastReportTime'] = pd.to_datetime(data['lastReportTime'], errors='coerce')
        data.set_index('lastReportTime', inplace=True)
        data = data.resample('d').mean()
        data = data.reset_index()

        xfmt = md.DateFormatter('%Y-%m-%d')
        plt.figure(figsize=(8,6))
        plt.xticks( rotation=25 )

        plt.plot(data['lastReportTime'][:14], data['p_sum'][:14], marker='.', label='Load')
        plt.title(label='Load Consumption on Weekend and Holiday')
        plt.legend(loc='upper left')
        plt.ylabel('Load(W)')
        plt.xlabel('Time')
        plt.gca().xaxis.set_major_formatter(xfmt)
        # plt.label(loc = 'upper right')
        # plt.title('Load - Processed Data')
        plt.show()

        # data = pd.read_csv('mtbuilding_cleanedwithinterpolation_202007-202104.csv')
        data = data[:720]
        plt.figure(figsize=(8,6))
        pagi = data[data['session'].isin(['2'])]
        siang = data[data['session'].isin(['3'])]
        malem = data[data['session'].isin(['0'])]
        midnight = data[data['session'].isin(['1'])]
        sore = data[data['session'].isin(['4'])]

        # pagi.head()
        plt.title(label='Distribution of Load Consumption in Different Session Time', size=14)
        plt.boxplot([pagi['p_sum'],siang['p_sum'],sore['p_sum'],malem['p_sum'],midnight['p_sum']], labels=['Morning','Afternoon','Evening','Night','Midnight'])
        plt.ylabel(ylabel='Load Consumption')

        mt1 = pd.read_csv('MT1(2020_07-2021_04)_interpolation.csv')
        mt2 = pd.read_csv('MT2(2020_07-2021_04)_interpolation.csv')
        mt3 = pd.read_csv('MT3(2020_07-2021_04)_interpolation.csv')
        mt4 = pd.read_csv('MT4(2020_07-2021_04)_interpolation.csv')
        mt5 = pd.read_csv('MT5(2020_07-2021_04)_interpolation.csv')
        
        fig = go.Figure(go.Scatter(y=mt1['p_sum'], x =mt1['lastReportTime'], name='MT1'))
        fig.add_scatter(y=mt2['p_sum'], x =mt1['lastReportTime'], name='MT2')
        fig.add_scatter(y=mt3['p_sum'], x =mt1['lastReportTime'] ,name='MT3')
        fig.add_scatter(y=mt4['p_sum'], x =mt1['lastReportTime'] ,name='MT4')
        fig.add_scatter(y=mt5['p_sum'], x =mt1['lastReportTime'] ,name='MT5')
        fig.update_layout(title_text='Power Consumption')
        pio.show(fig)

def main():
    df = pd.read_csv("hotaiLog_2020_07-08.csv", encoding='latin-1')
    df2 = pd.read_csv("hotaiLog_2020_0901-1025.csv", encoding='latin-1')
    df3 = pd.read_csv('hotaiLog_2020_1026-1217.csv', encoding='latin-1', error_bad_lines=False)
    df4 = pd.read_csv('hotaiLog_20201218-20210131.csv', encoding='latin-1', error_bad_lines=False)
    df5 = pd.read_csv('hotaiLog_20210201-20210315.csv', encoding='latin-1', error_bad_lines=False)
    df6 = pd.read_csv('hotaiLog_20210316-20210428.csv', encoding='latin-1', error_bad_lines=False)
    dataset = pd.concat([df,df2,df3,df4,df5,df6])
    dataset.describe()

    pre = Preprocessing()
    mt = pre.mt(dataset)
    mt 
    ch = pre.chang(dataset)
    ch
    dataana = pd.read_csv('mtbuilding_cleanedwithinterpolation_202007-202104.csv')
    ana = pre.analysis(dataana)
    ana
if __name__ == '__main__':
    main()