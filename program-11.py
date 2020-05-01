
# coding: utf-8

#!/bin/env python
'''
Date created: 4/20/2020
Author: Gargeya Vunnava
Github name: gargeyavunnava
Purdue username: vvunnava
Assignment 11: In this assignment, we take the annual and monthly statistics ofwildcat creek 
and tippecanoe and plot selective figures for a scientific presentation. The figs are stored in high resolution .svg format so that 
they can be enlarged later at higher quality if required (also, .svg is compatible with microsoft powerpoint).
'''

import pandas as pd
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    #removing negative discharge values as gross error check
    DataDF=DataDF[~(DataDF['Discharge']<0)]
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    #clipping the data between the dates given as function arguments
    DataDF=DataDF[startDate:endDate]
    MissingValues = DataDF["Discharge"].isna().sum()
      
    return( DataDF, MissingValues )

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    DataDF = pd.read_csv(fileName,parse_dates=['Date'], index_col=['Date'], delimiter = ',')

    return( DataDF )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
        
    MonthlyAverages = pd.DataFrame(index = range(1,13),columns = MoDataDF.columns)
    #sum all months across all years in show the mean data in a dataframe 'MonthlyAverages'
    for i in range(1,13):
        MonthlyAverages.iloc[i-1,:] = MoDataDF.loc[(MoDataDF.index.month == i)].mean()
        MonthlyAverages['Station'] = MoDataDF.iloc[0,-1]
    return( MonthlyAverages )

def plot_5yr_flow( tippe_df_5yrs, wildcat_df_5yrs ):
    # Reading the 5 year discharge rate values for both rivers to a dataframe
    df = pd.DataFrame({'Tippecanoe river': tippe_df_5yrs['Discharge'], 'Wildcat creek':wildcat_df_5yrs['Discharge']},index = tippe_df_5yrs.index)
    # The pandas inbuilt plot function is used to plot the line graphs
    fig1= df.plot(color=['k','r'],alpha=0.65,figsize = (10,5))
    fig1.set_ylabel('Daily flow rate (cubic ft/s)') # set y label
    fig1.set_xlabel('Date') # set x label
    fig1.set_title('Daily flow rates of Tippecanoe river and Wildcat creek between 15/3/2015 and 15/3/2020 (cubic ft/s)')
    return(plt.savefig('5_yr_flow.svg', dpi = 96, edgecolor = 'black')) # storing plots as .svg files with 96 dpi

def annual_plot( variable ):
    #This function takes in a variable as a string and plots the annual variable data for each river.
    df = pd.DataFrame({'Tippecanoe river': annual_tipp[variable], 'Wildcat creek':annual_wild[variable]},index = annual_wild.index)
    ax = df.plot(kind = 'line',
                   color=['k','y'],
                   alpha=0.65, # setting transparancy to see overlapping lines
                   figsize = (10,5)) #fig size in inches
    ax.legend(loc = 'upper center')
    ax.set_xlabel('Date')
    ax.set_ylabel(variable)
    ax.set_title('Annual '+variable+' data for Tippecanoe river and Wildcat creek.')
    
    return(plt.savefig(variable+'.svg', dpi = 96, edgecolor = 'black')) # storing plots as .svg files with 96 dpi


def plot_exceed_prob( annual_wild,annual_tipp ):
    #This function takes the annual data values and calculates the Return period of annual peak flow events for both rivers.
    # It plots a scatter plot between exceedence probability and flow rate for each river
    data = [annual_wild,annual_tipp]
    rank = [None]*2

    for i in range(2):
        df = data[i].sort_values('Peak Flow', ascending=False)['Peak Flow'].to_frame() #sorting flow rate in descending order
        N = len(df)
        df = df.reset_index(drop=True)
        df['Rank'] = range(1,N+1) #Initiating a 'Rank' column
        df['Exceedence Probability'] = 0 #Initiating a Exceedence Probabilitycolumn
        ex_prob = [0]*N
        for j in range(N):
            df.iloc[j,2] = (df.iloc[j,1]/(N+1)) # calculaitng Exceedence Probability for each flow instance
        rank[i] = df
    # Plotting Exceedence Probability     
    Y1 = rank[0]['Peak Flow'].tolist()
    Y2 = rank[1]['Peak Flow'].tolist()
    X1 = rank[0]['Exceedence Probability'].tolist()
    X2 = rank[1]['Exceedence Probability'].tolist()
    fig, ax = plt.subplots()
    fig.set_size_inches(10,5)
    ax.plot(X1,Y1,'o', label= 'Wildcat creek')
    ax.plot(X2,Y2,'x', label = 'Tippecanoe river')
    ax.legend(loc="upper left")
    ax.set_xlim(1,0)
    ax.set_xlabel('Exceedence Probability')
    ax.set_ylabel('Flow rate (cubic ft/s)')
    ax.set_title('Return period of annual peak flow events for Tippecanoe river and Wildcat creek.')
    return(plt.savefig('Exceedence Probability.svg', dpi = 96, edgecolor = 'black')) # storing plots as .svg files with 96 dpi

def plot_ave_annual_monthly_flow(monthly_wild, monthly_tipp):
    #This function plots the average annual monthly flow rate values for both the rivers
    data = [monthly_tipp, monthly_wild]
    
    df = pd.DataFrame(columns = ['Wildcat creek','Tippecanoe river']) #storing ave annual monthly flows of both rivers in a dataframe
    
    for i in range(2):
        df.iloc[:,i] = GetMonthlyAverages(data[i])['Mean Flow']
        
    fig1 = df.plot(color=['g','r'],alpha=0.65,figsize = (10,5)) #using pandas default plotting function
    fig1.set_ylabel('Average annual monthly flow (cubic ft/s)') # set y label
    fig1.set_xlabel('Month') # set x label
    fig1.set_title('Average annual monthly flow rate for Tippecanoe river and Wildcat creek (cubic ft/s)')
    return(plt.savefig('ave_annual_month_flow.svg', dpi = 96, edgecolor = 'black')) # storing plots as .svg files with 96 dpi



# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    '''The following code uses custom plotting functions defined previously in this file.

    '''
    #storing file names as variables
    annual = 'Annual_Metrics.csv' 
    monthly = 'Monthly_Metrics.csv'

    tippe = 'TippecanoeRiver_Discharge_03331500_19431001-20200315.txt'
    wildcat = 'WildcatCreek_Discharge_03335000_19540601-20200315.txt'

    annual_df = ReadMetrics( annual ) # storing annual metrics from previous assignment in a df
    monthly_df = ReadMetrics( monthly ) # storing monthly metrics from previous assignment in a df

    tippe_df = ReadData( tippe )[0] # storing data from .txt files in a df
    wildcat_df = ReadData( wildcat )[0] # storing data from .txt files in a df
    
    #Clipping data to last 5 yrs of available data
    tippe_df_5yrs = ClipData(tippe_df,'2015-03-15','2020-03-15')[0]
    wildcat_df_5yrs = ClipData(wildcat_df,'2015-03-15','2020-03-15')[0]

    annual_wild = annual_df.loc[annual_df.Station == 'Wildcat'] # df to store only annual wildcat data
    annual_tipp = annual_df.loc[annual_df.Station == 'Tippecanoe'] # df to store only annual tippe data

    monthly_wild = monthly_df.loc[monthly_df.Station == 'Wildcat'] # df to store only monthly wildcat data
    monthly_tipp = monthly_df.loc[monthly_df.Station == 'Tippecanoe'] # df to store only monthly wildcat data

    #plotting 5 year flow data for the 2 rivers using function defined above and saving the file as .svg
    plot_5yr_flow( tippe_df_5yrs, wildcat_df_5yrs )
    
    #plotting annual variable data using function defined above and saving the file as .svg
    annual_plot( 'Coeff Var' )
    annual_plot( 'Tqmean' )
    annual_plot( 'R-B Index' )
    
    #plotting ave annual monthly fow data using function defined above and saving the file as .svg
    plot_ave_annual_monthly_flow(monthly_wild, monthly_tipp)
    
    #plotting exccedence probability data using function defined above and saving the file as .svg
    plot_exceed_prob( annual_wild,annual_tipp )

