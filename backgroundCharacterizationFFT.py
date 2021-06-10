# -*- coding: utf-8 -*-
"""
Background removal process invlving 2D-FFT

Malachi Mooney-Rivkin
Last Edit: 6/3/2021
Idaho Space Grant Consortium
moon8435@vandals.uidaho.edu
"""

#dependencies
import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#for ellipse fitting
from math import atan2
from numpy.linalg import eig, inv, svd

#data smoothing
from scipy import signal

#metpy related dependencies - consider removing entirely
import metpy.calc as mpcalc
from metpy.units import units

#tk gui
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.font import Font
from tkinter import ttk

#skimage ellipse fitting
from skimage.measure import EllipseModel


###############################BEGINING OF USER INPUT##########################

#variables that are specific to analysis: These might be changed regularly depending on flight location, file format, etc.
flightData = r"C:\Users\Malachi\OneDrive - University of Idaho\%SummerInternship2020\%%CHIILE_Analysis_Backups\ChilePythonEnvironment_01112021\ChileData_012721\Tolten_01282021"             #flight data directory
fileToBeInspected = 'T26_1630_12142020_MT2.txt'                                                 #specific flight profile to be searched through manually
microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T26_all"  
#microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T28"              #location where selections from GUI ard. This is also the location where do analysis looks for micro hodos to analysis
waveParamDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory"     #location where wave parameter files are to be saved



##################################END OF USER INPUT######################

def preprocessDataResample(file, path, spatialResolution, lambda1, lambda2, order):
    #delete gloabal data variable as soon as troubleshooting is complete
    global data
    """ prepare data for hodograph analysis. non numeric values & values > 999999 removed, brunt-viasala freq
        calculated, background wind removed

        Different background removal techniques used: rolling average, savitsky-golay filter, nth order polynomial fits
    """
 
    #indicate which file is in progress
    print("Analyzing: {}".format(file))
    
    # Open file
    contents = ""
    f = open(os.path.join(path, file), 'r')
    print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Profile Data:":
            contents = f.read()  # Read in rest of file, discarding header
            print("File contains GRAWMET profile data")
            break
    f.close()  # Need to close opened file


    # Read in the data and perform cleaning
    # Need to remove space so Virt. Temp reads as one column, not two
    contents = contents.replace("Virt. Temp", "Virt.Temp")
    # Break file apart into separate lines
    contents = contents.split("\n")
    contents.pop(1)  # Remove units so that we can read table
    index = -1  # Used to look for footer
    for i in range(0, len(contents)):  # Iterate through lines
        if contents[i].strip() == "Tropopauses:":
            index = i  # Record start of footer
    if index >= 0:  # Remove footer, if found
        contents = contents[:index]
    contents = "\n".join(contents)  # Reassemble string

    # format flight data in dataframe
    data = pd.read_csv(StringIO(contents), delim_whitespace=True)
    
    #turn strings into numeric data types, non numerics turned to nans
    data = data.apply(pd.to_numeric, errors='coerce') 

    # replace all numbers greater than 999999 with nans
    data = data.where(data < 999999, np.nan)    

    #truncate data at greatest alt
    data = data[0 : np.where(data['Alt']== data['Alt'].max())[0][0]+1]  
    print("Maximum Altitude: {}".format(max(data['Alt'])))

    #drop rows with nans
    data = data.dropna(subset=['Time', 'T', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt'])
    
    #remove unneeded columns
    data = data[['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.']]
    
    #linearly interpolate data - such that it is spaced iniformly in space, heightwise - stolen from Keaton
    #create index of heights with 1 m spacial resolution - from minAlt to maxAlt
    heightIndex = pd.DataFrame({'Alt': np.arange(min(data['Alt']), max(data['Alt']))})
    #right merge data with index to keep all heights
    data= pd.merge(data, heightIndex, how='right', on='Alt')
    #sort data by height
    data = data.sort_values(by='Alt')
    #linear interpolate the nans
    missingDataLimit = 999  #more than 1km of data should be left as nans, will not be onsidered in analysis
    data = data.interpolate(method='linear', limit=missingDataLimit)
    #resample at height interval
    keepIndex = np.arange(0, len(data['Alt']), spatialResolution)
    data = data.iloc[keepIndex,:]
    data.reset_index(drop=True, inplace=True)
    
    #change data container name, sounds silly but useful for troubleshooting data-cleaning bugs
    global df
    df = data
    #print(df)
    #make following vars availabale outside of function - convenient for time being, but consider changing in future
    """
    global Time 
    global Pres 
    global Temp 
    global Hu 
    global Wd 
    global Long 
    global Lat 
    global Alt 
    global potentialTemp
    global bv2
    global u, v 
    global uBackground 
    global vBackground
    global tempBackground

    #for comparing rolling ave to savitsky golay
    #global uBackgroundRolling
    #global vBackgroundRolling
    #global tempBackgroundRolling
    #global uRolling
    #global vRolling
    #global tRolling
    
    """
    #individual series for each variable, local
    Time = df['Time'].to_numpy()
    Pres = df['P'].to_numpy() * units.hPa
    Temp = df['T'].to_numpy()  * units.degC
    Ws = df['Ws'].to_numpy() * units.m / units.second
    Wd = df['Wd'].to_numpy() * units.degree
    Long = df['Long.'].to_numpy()
    Lat = df['Lat.'].to_numpy()
    Alt = df['Alt'].to_numpy().astype(int) * units.meter
    
    
    #calculate brunt-viasala frequency **2 
    tempK = Temp.to('kelvin')
    potentialTemperature =  tempK * (p_0 / Pres) ** (2/7)    #https://glossary.ametsoc.org/wiki/Potential_temperature   
    bv2 = mpcalc.brunt_vaisala_frequency_squared(Alt, potentialTemperature).magnitude    #N^2 
    #bv2 = bruntViasalaFreqSquared(potentialTemperature, heightSamplingFreq)     #Maybe consider using metpy version of N^2 ? Height sampling is not used in hodo method, why allow it to affect bv ?
    
    #convert wind from polar to cartesian c.s.
    u, v = mpcalc.wind_components(Ws, Wd)   #raw u,v components - no different than using trig fuctions
    print("Size of u: ", len(u))
    #subtract nth order polynomials to find purturbation profile
    
        
    return 

def f(x,y):
    """
    invent function to test plotting with
    """
    t=3
    w=1
    p = 1 #1 #5 #2 #5
    q = 2
    return np.sin(x) * np.sin(y)


def constructBackGroundFFT(directory):
    for file in os.listdir(directory):
        #print(file)
        a=1
        
    print("HERE")
    xx = np.linspace(0,10*np.pi)
    yy = np.linspace(0,5*np.pi)
    xv, yv = np.meshgrid(xx, yy)
    print("length meshgrid: ", len(xv), "Length y:", len(yv))
    global z
    z = f(xv,yv)
    print(z)
    plt.imshow(z, interpolation='nearest')
    
    import scipy as sp
    fft = sp.fft.fft2(z)
    global freqs
    ###
    FreqCompRows = np.fft.fftfreq(FFTData.shape[0],d=2) # from internet...
    FreqCompCols = np.fft.fftfreq(FFTData.shape[1],d=2)
    ###
    freqs = sp.fft.fftfreq(np.size(z))
    print("Freqs: ", freqs)
    print("FFT: ", fft)
    
    
    fig, ax = plt.subplots()
    ax.contourf(fft)
    #xy = np.column_stack([xv,yv,z])
    #ax.contourf(xv,yv,z)
    #xy_fft = np.fft.fft2(xy)
    
    
    return

#Run data to construct background
constructBackGroundFFT(flightData)