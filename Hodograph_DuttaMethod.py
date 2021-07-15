# -*- coding: utf-8 -*-
"""
Methods adopted form Dutta (2017)

Manual Hodograph Analyzer
Include this script in the same folder as the input and output directories, or else specify their file path

Semantics:
Enclosed structures contained in full-flight hodographs are reffered to as "microhodographs", 
let u_o = (U_x0, U_y0, 0) be the background wind and u_1 = (u, v, w) be the perturbed velocities 

To Do:
- add/update sources for parameter calculations
- format parameter output to match wavelet python code? Maybe doesnt make sense to use json, as profile data needs to accompany wave parameters for plotting
- get rid of metpy library?
- clean up bulk hodograph plots
- add time to wave parameter list
    

Malachi Mooney-Rivkin
Last Edit: 6/2/2021
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

#Which functionality would you like to use?
showVisualizations = False     # Displays macroscopic hodograph for flight
siftThruHodo = True    # Use manual GUI to locate ellipse-like structures in hodograph
analyze = False   # Display list of microhodographs with overlayed fit ellipses as well as wave parameters
applyButterworth = True #should Butterworth filter be applied to data? Linear interpolation is also implemented, prior to filtering, at specified spatial resolution
location = "Tolten"     #[Tolten]/[Villarica]

user = r'\Malachi'
#variables that are specific to analysis: These might be changed regularly depending on flight location, file format, etc.
flightData = r"C:\Users\reec7164\OneDrive - University of Idaho\Eclipse\Data\Tolten"             #flight data directory
fileToBeInspected = 'Tolten_L30_121420_UTC2030.txt'                                                 #specific flight profile to be searched through manually
microHodoDir = r"C:\Users\reec7164\OneDrive - University of Idaho\Eclipse\Hodographs\MicroHodos"
#microHodoDir = r"C:\Users\Malachi\OneDrive - University of Idaho\workingChileDirectory\Tolten\T28"              #location where selections from GUI ard. This is also the location where do analysis looks for micro hodos to analysis
waveParamDir = r"C:\Users\reec7164\OneDrive - University of Idaho\Eclipse\Hodographs\Parameters"     #location where wave parameter files are to be saved

#for Kathryn:
#flightData = r"C:\Users\reec7164\OneDrive - University of Idaho\%SummerInternship2020\%%CHIILE_Analysis_Backups\ChilePythonEnvironment_01112021\ChileData_012721\Tolten_01282021"             #flight data directory
#fileToBeInspected = 'T36_0230_121520_Artemis_Rerun_CLEAN.txt'                                                 #specific flight profile to be searched through manually
#microHodoDir = r"C:\Users\Moon8435\OneDrive - University of Idaho\workingChileDirectory\T36_hodographs"
#microHodoDir = r"C:\Users\reec7164\OneDrive - University of Idaho\workingChileDirectory\Tolten\T28"              #location where selections from GUI ard. This is also the location where do analysis looks for micro hodos to analysis
#waveParamDir = r"C:\Users\reec7164\OneDrive - University of Idaho\Eclipse\Hodographs\Parameters"     #location where wave parameter files are to be saved

if location == "Tolten":
    latitudeOfAnalysis = abs(-39.236248) * units.degree    #latitude at which sonde was launched. Used to account for affect of coriolis force.
elif location == "Villarica":
    latitudeOfAnalysis = abs(-39.30697) * units.degree     #same, but for Villarica...

g = 9.8                     #m * s^-2
spatialResolution = 5
heightSamplingFreq = 1/spatialResolution      #1/m used in interpolating data height-wise
minAlt = 1000 * units.m     #minimun altitude of analysis
p_0 = 1000 * units.hPa      #needed for potential temp calculatiion
n_trials = 1000         #number of bootstrap iterations
#for butterworth filter
lowcut = 500  #m - lower vertical wavelength cutoff for Butterworth bandpass filter
highcut = 4000  #m - upper vertical wavelength cutoff for Butterworth bandpass filter
order = 3   #Butterworth filter order - Dutta(2017)
#modes for data preprocessing
backgroundPolyOrder = 3
applyButterworth = True
tropopause = 11427  #m


##################################END OF USER INPUT######################
scriptDirectory = os.getcwd()
print(scriptDirectory)

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
    #print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Profile Data:":
            contents = f.read()  # Read in rest of file, discarding header
            #print("File contains GRAWMET profile data")
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
    
    #Truncate data below tropopause
    data = data[data['Alt'] >= tropopause] 
    print("Minimum Altitude: {}".format(min(data['Alt'])))

    #drop rows with nans
    data = data.dropna(subset=['Time', 'T', 'Ws', 'Wd', 'Long.', 'Lat.', 'Alt'])
    
    #remove unneeded columns
    data = data[['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.']]
    global tempData
    tempData = data
    if applyButterworth:
        #linear interpolation only needs to occur if butterworth is applied
        
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

    plt.plot(Alt,bv2)
    meanBV2 = np.ones(len(bv2)) * np.mean(bv2)
    # localmean =
    plt.plot(Alt, meanBV2)
    #bv2 = bruntViasalaFreqSquared(potentialTemperature, heightSamplingFreq)     #Maybe consider using metpy version of N^2 ? Height sampling is not used in hodo method, why allow it to affect bv ?
    

    #convert wind from polar to cartesian c.s.
    u, v = mpcalc.wind_components(Ws, Wd)   #raw u,v components - no different than using trig fuctions

    #subtract nth order polynomials to find purturbation profile
    #detrend  temperature using polynomial fit
    '''
    Fig, axs = plt.subplots(2,4,figsize=(6,6), num=1)   #figure for temperature
    Fig2, axs2 = plt.subplots(2,4,figsize=(6,6), num=2)   #figure for wind
    axs = axs.flatten() #make subplots iteratble by single indice
    axs2 = axs2.flatten()
    '''
    temp_background = []
    u_background = []
    v_background = []

    for k in range(2,10):
        i = k-2
        
        #temp
        poly = np.polyfit(Alt.magnitude / 1000, Temp.magnitude, k)
        tempFit = np.polyval(poly, Alt.magnitude / 1000)
        temp_background.append(tempFit)
        
        #u
        poly = np.polyfit(Alt.magnitude / 1000, u.magnitude, k)
        uFit = np.polyval(poly, Alt.magnitude / 1000)
        u_background.append(uFit)

        #v
        poly = np.polyfit(Alt.magnitude / 1000, v.magnitude, k)
        vFit = np.polyval(poly, Alt.magnitude / 1000)
        v_background.append(vFit)
        '''
        #plot temp
        axs[i].plot(tempFit, Alt.magnitude / 1000, color='darkblue')
        axs[i].plot(Temp.magnitude, Alt.magnitude / 1000)
        axs[i].set_title("Order: " + str(k))
        #axs[i].set_xlabel("Temperature (C)")
        axs[i].set_ylabel("Altitude (km)")
        axs[i].tick_params(top=True, right=True)
        
        #plot u
        zonal, = axs2[i].plot(uFit, Alt.magnitude / 1000, color='darkblue', label='Zonal')
        axs2[i].plot(u.magnitude, Alt.magnitude / 1000, color='darkblue')
        axs2[i].set_title("Order: " + str(k))
        #axs[i].set_xlabel("Temperature (C)")
        axs2[i].set_ylabel("Altitude (km)")
        axs2[i].tick_params(top=True, right=True)
        
        #plot v
        meridional, = axs2[i].plot(vFit, Alt.magnitude / 1000, color='darkred', label='Meridional')
        axs2[i].plot(v.magnitude, Alt.magnitude / 1000, color='darkred')
        #axs2[i].set_title("Order: " + str(k))
        #axs[i].set_xlabel("Temperature (C)")
        #axs2[i].set_ylabel("Altitude (km)")
        #axs2[i].tick_params(top=True, right=True)
        '''
        
    '''    
    #Fig - hide labels and ticks, add notations
    plt.figure(1)
    Fig.add_subplot(111, frameon=False) #make hidden subplot to add xlabel
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False) #which='both'
    plt.xlabel("Temperature (C)")
    Fig.suptitle("Temperature; Polynomial Fitted \n {}".format(file))
    for ax in axs:
        ax.label_outer()
    
    #Fig2 - hide labels and ticks, add notations
    plt.figure(2)
    Fig2.suptitle("Wind; Polynomial Fitted \n {}".format(file))
    for ax in axs2:
        ax.label_outer()
    Fig2.add_subplot(111, frameon=False) #make hidden subplot to add xlabel
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False) #which='both'
    plt.xlabel("Winds (m/s)")
    #handles, labels = axs2.get_legend_handles_labels()
    #Fig2.legend(handles,labels, loc='lower center')
    Fig2.legend(handles=[zonal, meridional], labels=['Zonal', 'Meridional'], loc='lower center', ncol=2)
    '''
    
    #subtract fits to produce various perturbation profiles
    tempPert = []
    global uPert
    uPert = []
    vPert = []
    
    for i, element in enumerate(temp_background):
        pert = np.subtract(Temp.magnitude, temp_background[i])
        tempPert.append(pert)
        pert = np.subtract(u.magnitude, u_background[i])
        uPert.append(pert)
        pert = np.subtract(v.magnitude, v_background[i])
        vPert.append(pert)
        
    
    #plot to double check subtraction
    Fig, axs = plt.subplots(2,2,figsize=(6,6), num=3, sharey=True)#, sharex=True)   #figure for u,v butterworth filter
    for i,element in enumerate(u_background):
        axs[0,0].plot(uPert[i], Alt.magnitude/1000, linewidth=0.5, label="Order: {}".format(str(i+2)))
        axs[1,0].plot(vPert[i], Alt.magnitude/1000, linewidth=0.5)   
   
    Fig.legend()
    Fig.suptitle("Wind Components; Background Removed, Filtered \n {}".format(file))
    axs[0,0].set_xlabel("Zonal Wind (m/s)")
    axs[1,0].set_xlabel("Meridional Wind (m/s)")
    axs[0,1].set_xlabel("Filtered Zonal Wind (m/s)")
    axs[0,1].set_xlim([-10,10])
    axs[1,1].set_xlim([-10,10])
    axs[0,0].set_xlim([-20,35])
    axs[1,0].set_xlim([-20,35])
    axs[1,1].set_xlabel("Filtered Meridional Wind (m/s)")
    axs[0,0].set_ylabel("Altitude (km)")
    axs[1,0].set_ylabel("Altitude (km)")
    axs[0,0].tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off
    axs[0,1].tick_params(axis='x',labelbottom=False) # labels along the bottom edge are off


    #Apply Butterworth Filter
    if applyButterworth:
        #filter using 3rd order butterworth - fs=samplerate (1/m)
        freq2 = 1/lambda1    #find cutoff freq 1/m
        freq1 =  1/lambda2    #find cutoff freq 1/m
        
        # Plot the frequency response for a few different orders.
        b, a = butter_bandpass(freq1, freq2, heightSamplingFreq, order)
        w, h = signal.freqz(b, a, worN=5000)
        plt.figure(4)
        plt.plot(w/np.pi, abs(h))
        plt.plot([0, 1], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(1/2)')
        plt.xlabel('Normalized Frequency (x Pi rad/sample) \n [Nyquist Frequency = 1]') #1/m ?
        plt.ylabel('Gain')
        plt.xlim([0,.1])
        plt.grid(True)
        plt.title("Frequency Response of 3rd Order Butterworth Filter \n Vertical Cut-off Wavelengths: 0.5 - 4 km")
        plt.legend(loc='best')
    
        # Filter a noisy signal.
        uButter = []
        vButter = []
        tempButter = []
        for i,element in enumerate(vPert):
            
            filtU = butter_bandpass_filter(uPert[i],freq1, freq2, heightSamplingFreq, order)
            uButter.append(filtU)
            filtV = butter_bandpass_filter(vPert[i], freq1, freq2, 1/5, order)
            vButter.append(filtV)
            filtTemp = butter_bandpass_filter(tempPert[i],freq1, freq2, heightSamplingFreq, order)
            tempButter.append(filtTemp)
            #axs[1,1].plot(vPert[0], Alt.magnitude)
            axs[1,1].plot(vButter[i], Alt.magnitude/1000, linewidth=0.5)
            axs[0,1].plot(uButter[i], Alt.magnitude/1000, linewidth=0.5)
            #plt.xlabel('time (seconds)')
            #plt.hlines([-a, a], 0, T, linestyles='--')
            #plt.grid(True)
            #plt.axis('tight')
            #plt.legend(loc='upper left')
    
    
    
        #re define u,v
        u = uButter[4]
        v = vButter[4]
        Temp = tempButter[4]
        print("Butterworth Filter Applied")
    
    #polyIndice = backgroundPolyOrder - 2
    #u = uPert[polyIndice] * units.m / units.second
    #v = vPert[polyIndice] * units.m / units.second
    #Temp = tempPert[polyIndice] * units.degC

    processedData = []
    return processedData

def butter_bandpass(lowcut, highcut, fs, order):
    """
        Used for plotting the frequency response of Butterworth
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """
        Applies Butterworth filter to perturbation profiles
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def bruntViasalaFreqSquared(potTemp, heightSamplingFreq):
    """ replicated from Tom's script
    """
    G = 9.8 * units.m / units.second**2
    N2 = (G / potTemp) * np.gradient(potTemp, heightSamplingFreq * units.m)     #artifact of tom's code, 
    return N2

def unitCirc2Azmith(radians):
    """Calculate compass direction in degrees, input: unit circle angle in radians
    """
    degrees = np.rad2deg(radians)
    degrees = 450 - degrees    #rotate cordinate system 90 deg CCW
    while degrees < 0:
        degrees += 360
    if degrees > 360:
        degrees %= 360
    return degrees

class microHodo:
    def __init__(self, ALT, U, V, TEMP, BV2, LAT, LONG, TIME, ORIENTATION):
      self.alt = ALT#.magnitude
      self.u = U#.magnitude
      self.v = V#.magnitude
      self.temp = TEMP#.magnitude
      self.bv2 = BV2#.magnitude
      self.lat = LAT
      self.long = LONG
      self.time = TIME
      self.orientation = ORIENTATION
      #self.orientation = np.full((len(self.time), 1), ORIENTATION)
      
    def addOrientation(self, ORIENTATION):
        self.orientation = np.full((len(self.time), 1), ORIENTATION)
      
      
    def addNameAddPath(self, fname, fpath):
        #adds file name attribute to object
        self.fname = fname
        self.savepath = fpath
        
    def addAltitudeCharacteristics(self):
        self.lowerAlt = min(self.alt).astype('int')
        self.upperAlt = max(self.alt).astype('int')
      
    

    def saveMicroHodoNoIndices(self):
        """ dumps microhodograph object attributs into csv 
        """
    
        T = np.column_stack([self.time, self.alt.magnitude, self.u.magnitude, self.v.magnitude, self.temp.magnitude, self.bv2, self.lat, self.long, self.orientation]) 
        #T = np.column_stack([self.time, self.alt, self.u, self.v, self.temp, self.bv2, self.lat, self.long, self.orientation])
        T = pd.DataFrame(T, columns = ['time', 'alt', 'u', 'v', 'temp', 'bv2', 'lat','long', 'orientation'])
        
        
        
        fname = '{}_microHodograph_{}-{}'.format(self.fname.strip('.txt'), int(self.alt[0].magnitude), int(self.alt[-1].magnitude))
        T.to_csv('{}/{}.csv'.format(self.savepath, fname), index=False)                          

    def fit_ellipse(self):
            """Fitting algorithm; approximates local hodograph, theta smallest angle in radians from +u axis
            """
            points = np.array([self.u, self.v])
            ell = EllipseModel()
            ell.estimate(points.transpose())
            xc, yc, a, b, theta = ell.params
            print("Unmodified Theta: ", theta)
            do = True
            if a<b and do:
                print("Swappring a,b etc")
                #swap a, b if in wrong order
                a,b = b,a
                theta = theta - np.pi/2

            if theta > np.pi/2:
                     theta = theta - np.pi
            if theta < -np.pi/2:
                 theta = theta + np.pi







            # if theta < 0:
            #     theta = 2 * np.pi + theta
            #
            # if theta > (2 * np.pi):
            #     theta = theta % (2 * np.pi)

            self.a = a
            self.b = b
            self.c_x = xc
            self.c_y = yc
            self.phi = theta    # radians, from +u' axis
            print("Ellipse fit theta: ", np.rad2deg(theta))
            print("Ellipse fit theta: ", theta)
            return a, b, xc, yc, theta
    
    def bootstrap_params(self, n_trials):
        """
        Iterate through a variety of random samplings, testing the ellipse fit of each sample
        """
        uu, vv = np.asarray(self.u), np.asarray(self.v)
        #arrays to store parameters of all fits
        aas, bs, c_xs, c_ys, phis = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)
        
        a, b, c_x, c_y,phi = self.fit_ellipse() # first trial
        aas[0] = a
        bs[0] = b
        c_xs[0] = c_x
        c_ys[0] = c_y
        phis[0] = phi
        
 
        num_data_points = self.u.shape[0] # or len(u), depending on if it's an array or list
        
        #keep track of erroneous fits
        badIndices = []
        for i in range(1, n_trials):
 
            random_indices = np.random.choice(num_data_points, size=num_data_points,replace=True) # grab indices with repetition to subsample your data
            random_subset_u = uu[random_indices] # make sure u and v are numpy arrays
            random_subset_v = vv[random_indices] # with u = np.asarray(u) if u is a list
            self.u = random_subset_u
            self.v = random_subset_v
            #print("random indices: ", random_indices)                               
            a, b, c_x, c_y,phi = self.fit_ellipse() # get estimate of params for this bootstrapped sample
                
                
            if np.isnan(a) | np.isnan(b): 
                badIndices.append(i)
            else:                           
                aas[i] = a
                bs[i] = b
                c_xs[i] = c_x
                c_ys[i] = c_y
                phis[i] = phi
         
            #return u,v to unperturbed state
            self.u = uu
            self.v = vv
        
            
        #get rid of iterations with bad indices
        print("bad indices: ", len(badIndices))
        aas = np.delete(aas, badIndices)
        bs = np.delete(bs, badIndices)
        c_xs = np.delete(c_xs, badIndices)
        c_ys = np.delete(c_ys, badIndices)
        phis = np.delete(phis, badIndices)
        
        sdAR = np.std(aas/bs)
        
        #create bootstrap values-means
        self.bsa = np.nanmean(aas)
        self.bsb = np.nanmean(bs)
        self.bsc_x = np.nanmean(c_xs)
        self.bsc_y = np.nanmean(c_ys)
        self.bsphi = np.nanmean(phis)
        
        
        print("a,b,x,y,phi: ", self.bsa,self.bsb,self.bsc_x,self.bsc_y,self.bsphi)
        print("STDev AR: ", sdAR)
        
        return np.mean(aas), np.std(aas), np.mean(bs), np.std(bs), np.mean(c_xs), np.std(c_xs), np.mean(c_ys), np.std(c_ys), np.mean(phis), np.std(phis)# return statistics

    def getParameters(self):

        # Altitude of detection - mean
        self.altOfDetection = np.mean(self.alt)     # (meters)

        # Latitude of Detection - mean
        self.latOfDetection = np.mean(self.lat)     # (decimal degrees) 

        # Longitude of Detection - mean
        self.longOfDetection = np.mean(self.long)     # (decimal degrees)

        # Date/Time of Detection - mean - needs to be added!
        
        # Axial ratio
        wf = (self.a) / (self.b)    #long axis / short axis
        
        # Vertical wavelength
        self.lambda_z = self.alt[-1] - self.alt[0]       # (meters) -- Toms script multiplies altitude of envelope by two? 
        self.m = 2 * np.pi / self.lambda_z      # vertical wavenumber (rad/meters)

        # Horizontal wavelength
        bv2Mean = np.mean(self.bv2)
        coriolisFreq = mpcalc.coriolis_parameter(latitudeOfAnalysis)
        
        # test to determine if k_h is imaginary
        radical = (coriolisFreq.magnitude**2 * self.m**2) / abs(bv2Mean) * (wf**2 - 1)
        if radical < 0:
            print("WARNING: negative value encountered in radical")
        try:
         k_h = np.sqrt((coriolisFreq.magnitude**2 * self.m**2) / abs(bv2Mean) * (wf**2 - 1)) #horizontal wavenumber (1/meter)
        except:
            print("\033[1;31;40m Runtime warning encountered  \n")
        self.lambda_h = 1 / k_h     #horizontal wavelength (meter) #should be 2pi/k

        #Propogation Direction (Marlton 2016)
        #rot = np.array([[np.cos(self.phi), -np.sin(self.phi)], [np.sin(self.phi), np.cos(self.phi)]])       #2d rotation matrix - containinng angle of fitted elipse - as used in Toms script
        rot = np.array([[np.cos(-self.phi), -np.sin(-self.phi)], [np.sin(-self.phi), np.cos(-self.phi)]])       #2d rotation matrix - containinng  negative angle of fitted elipse
        uv = np.array([self.u, self.v])       #zonal and meridional components
        uvrot = np.matmul(rot,uv)       #change of coordinates
        urot = uvrot[0,:]               #urot aligns with major axis
        self.uRot = urot
        self.vRot = uvrot[1,:]
        print('UROT MAX', max(urot))
        dTdz = np.diff(self.temp)  / np.diff(self.alt)

        eta = np.mean(dTdz * urot[0:-1])

        if eta < 0:                 # check to see if temp perterbaton has same sign as u perterbation - clears up 180 deg ambiguity in propogation direction
            self.phi += np.pi
            print("Direction of orientaion reversed")

        self.directionOfPropogation = unitCirc2Azmith(self.phi) #get aorientation in degrees CW from North


        
        #Intrinsic horizontal phase speed (m/s)
        intrinsicFreq = coriolisFreq.magnitude * wf     #one ought to assign units to output from ellipse fitting to ensure dimensional accuracy
        intrinsicHorizPhaseSpeed = intrinsicFreq / k_h

        #extraneous calculations - part of Tom's script
        #k_h_2 = np.sqrt((intrinsicFreq**2 - coriolisFreq.magnitude**2) * (self.m**2 / abs(bv2Mean)))
        #int2 = intrinsicFreq / k_h_2

        #print("m: {}, lz: {}, h: {}, bv{}".format(self.m, self.lambda_z, intrinsicHorizPhaseSpeed, bv2Mean))
        #return altitude of detection, latitude, longitude, vertical wavelength,horizontal wavenumber, intrinsic horizontal phase speed, axial ratio l/s
        print("Parameters Calculated")

        return  [self.altOfDetection, self.lat[0], self.long[0], self.lambda_z/1000, self.lambda_h/1000, k_h, intrinsicFreq,  intrinsicHorizPhaseSpeed, wf, self.directionOfPropogation]
"""
    def improveFit(self):
        uu, vv = np.asarray(self.u), np.asarray(self.v)
        #get initial estimate of parameters
        a, b, c_x_old, c_y_old,phi = self.fit_ellipse() # get estimate of params
        
        #estimate center
        c_x_temp = np.mean(self.u)
        c_y_temp = np.mean(self.v)
        
        #subtract mean - this centers hodograph at origin
        self.u -= c_x_old
        self.v -= c_y_old
        
        #rotate ellipse to align major axis with u
        rot = np.array([[np.cos(-self.phi), -np.sin(-self.phi)], [np.sin(-self.phi), np.cos(-self.phi)]])       #2d rotation matrix - containinng  negative angle of fitted elipse
        uv = np.array([self.u, self.v])       #zonal and meridional components
        uvrot = np.matmul(rot,uv)       #change of coordinates
        self.urot = uvrot[0,:]               #urot aligns with major axis
        self.vrot= uvrot[1,:]
        self.u = uvrot[0,:]               #urot aligns with major axis
        self.v= uvrot[1,:]
        
        
        
        #scale minor axis by std of major axis this is fairly abitrary
        stdev_x = np.std(self.u)
        stdev_y = np.std(self.v)
        ratioOfEccentricity = stdev_x/stdev_y
        #ratioOfEccentricity = a/b
        self.v = self.v * ratioOfEccentricity
        
        #keep track of progression...
        self.uMod = self.u
        self.vMod = self.v
        
        #fit ellipse to rotated data
        a, b, c_x_new, c_y_new, phi = self.fit_ellipse() # get estimate of params
        self.arot = a
        self.brot = b
        self.c_x_improved = c_x_new+c_x_old
        self.c_y_improved = c_y_new+c_y_old
        self.phirot = phi
        self.c_x_rot = c_x_new
        self.c_y_rot = c_y_new
        
        
        #scale parameters back
        self.b_improved = b / ratioOfEccentricity
        self.a_improved = a
        
        #return u,v t original state
        self.u = uu
        self.v = vv
        a, b, c_x, c_y,phi = self.fit_ellipse() # get estimate of params
           
"""


def doAnalysis(microHodoDir):
    """ Extracts wave parameters from microHodographs; this function can be run on existing microhodograph files without needing to operate the GUI
    """
    #make sure files are retrieved from correct directory; consider adding additional checks to make sure user is querying correct directory
    print("Micro Hodograph Path Exists: ", os.path.exists(microHodoDir))
    
    hodo_list = []
    parameterList = []
    for file in os.listdir(microHodoDir):
        path = os.path.join(microHodoDir, file)
        print('Analyzing micro-hodos for flight: {}'.format(file))
        
        #dataframe from local hodograph file
        dtypes = "float, float, float, float, float, float, float, float, str"
        df = np.genfromtxt(fname=path, delimiter=',', names=True, dtype=None)

        #create microhodograph object, then start giving it attributes
        instance = microHodo(df['alt'], df['u'], df['v'], df['temp'], df['bv2'], df['lat'], df['long'], df['time'], df['orientation'][0])
        print("doAnalysis orientation: ", type(df['orientation'][0]))
        #file name added to object attribute here to be used in labeling plots
        instance.addNameAddPath(file, microHodoDir)  

        #find out min/max altitudes file
        instance.addAltitudeCharacteristics()

        #lets try to fit an ellipse to microhodograph
        #instance.bootstrap_params(n_trials)
        instance.fit_ellipse()

        #use ellipse to extract wave characteristics
        params = instance.getParameters()
        print("Wave Parameters: \n", params)

        #update running list of processed hodos and corresponding parameters
        parameterList.append(params)
        hodo_list.append(instance)  #add micro-hodo to running list
        print("")

    #organize parameters into dataframe; dump into csv


    parameterList = pd.DataFrame(parameterList, columns = ['Alt. [km]', 'Lat', 'Long', 'Vert Wavelength [km]', 'Horiz. Wavelength [km]', 'Horizontal Wave#', 'IntHorizPhase Speed', 'Int. Freq.', 'Axial Ratio L/S', 'Propagation Direction' ])


    parameterList.sort_values(by='Alt. [km]', inplace=True)
    

    pathAndFile = "{}\{}_params.csv".format(waveParamDir, fileToBeInspected.strip(".txt"))
    parameterList.to_csv(pathAndFile, index=False, na_rep='NaN')
    
    #sort list of hodographs in order of ascending altitude 
    hodo_list.sort(key=lambda x: x.altOfDetection)  

    return hodo_list     
    
def plotBulkMicros(hodo_list, fname):
    
    """ plot microhodographs in grid of subplots
    """ 
    """ This block of code is useful for inspecting single hodographs and their bestfit parameters
    #plots all micro-hodographs for a single flight
    bulkPlot = plt.figure(fname, figsize=(8.5,11))
    plt.suptitle("Micro-hodographs for \n {}".format(fname))#, y=1.09)
    
    totalPlots = len(hodo_list)
    if totalPlots > 20:
        bulkplot2 = plt.figure(fname+" pt2", figsize=(8.5,11))
    
    if totalPlots > 0:
        
        #figure out how to arrang subplots on grid
        numColumns = 4
        numRows = 5
        position = range(1, totalPlots + 1)
        
        
        i = 0   #counter for indexing micro-hodo objects
        for hodo in hodo_list:
            print("HODO ITERATION: ", hodo)
            #ax = bulkPlot.add_subplot(numRows, numColumns, position[i])#, aspect='equal')
            fig, ax = plt.subplots(figsize=(8.5,8.5))
            ax.plot(hodo_list[i].u, hodo_list[i].v, color='black', label='hodograph') 
            #axs[i].plot(hodo_list[i].u, hodo_list[i].v)
            
            #plot rotated raw data
            #ax.plot(hodo_list[i].urot, hodo_list[i].vrot, color='black', label='rotated', linewidth=1) 
            #plot scaled raw data
            #ax.plot(hodo_list[i].uMod, hodo_list[i].vMod, color='blue', label='scaled', linewidth=1)
        
            #plot parametric best fit ellipse
            param = np.linspace(0, 2 * np.pi)
            #best fit
            x = hodo_list[i].a * np.cos(param) * np.cos(hodo_list[i].phi) - hodo_list[i].b * np.sin(param) * np.sin(hodo_list[i].phi) + hodo_list[i].c_x
            y = hodo_list[i].a * np.cos(param) * np.sin(hodo_list[i].phi) + hodo_list[i].b * np.sin(param) * np.cos(hodo_list[i].phi) + hodo_list[i].c_y
            ax.plot(x, y, color='red', label='single fit')
            #axs[i].plot(x, y, color='red', label='single fit')
            
            #rotated fit
            #x = hodo_list[i].arot * np.cos(param) * np.cos(hodo_list[i].phirot) - hodo_list[i].brot * np.sin(param) * np.sin(hodo_list[i].phirot) + hodo_list[i].c_x_rot
            #y = hodo_list[i].arot * np.cos(param) * np.sin(hodo_list[i].phirot) + hodo_list[i].brot * np.sin(param) * np.cos(hodo_list[i].phirot) + hodo_list[i].c_y_rot
            #ax.plot(x, y, color='blue', label='rotated fit')
            
            #improved fit
            #x = hodo_list[i].a_improved * np.cos(param) * np.cos(hodo_list[i].phi) - hodo_list[i].b_improved * np.sin(param) * np.sin(hodo_list[i].phi) + hodo_list[i].c_x_improved
            #y = hodo_list[i].a_improved * np.cos(param) * np.sin(hodo_list[i].phi) + hodo_list[i].b_improved * np.sin(param) * np.cos(hodo_list[i].phi) + hodo_list[i].c_y_improved
            #ax.plot(x, y, color='purple', label='improved fit')
            #axs[i].plot(x, y, color='red', label='single fit')
            
            #try using skimage to fit
            xs = hodo_list[i].u
            ys = hodo_list[i].v
            a_points = np.array([xs, ys])
            print("SHAPE OF SKIMAGE ARRAY: ", np.shape(a_points.transpose()))
            ell = EllipseModel()
            ell.estimate(a_points.transpose())
            xc_skimage, yc_skimage, a_skimage, b_skimage, theta_skimage = ell.params
            
            #improved fit
            x = a_skimage * np.cos(param) * np.cos(theta_skimage) - b_skimage * np.sin(param) * np.sin(theta_skimage) + xc_skimage
            y = a_skimage * np.cos(param) * np.sin(theta_skimage) + b_skimage * np.sin(param) * np.cos(theta_skimage) + yc_skimage
            ax.plot(x, y, color='orange', label='improved fit')
            
            
            #bootstrapped values
            #x = hodo_list[i].bsa * np.cos(param) * np.cos(hodo_list[i].bsphi) - hodo_list[i].bsb * np.sin(param) * np.sin(hodo_list[i].bsphi) + hodo_list[i].bsc_x
            #y = hodo_list[i].bsa * np.cos(param) * np.sin(hodo_list[i].bsphi) + hodo_list[i].bsb * np.sin(param) * np.cos(hodo_list[i].bsphi) + hodo_list[i].bsc_y
            #ax.plot(x, y, color='blue', label='bootstrapped fit')
            
            ax.set_xlabel("(m/s)", fontsize=7)
            ax.set_ylabel("(m/s)", fontsize=7)
            ax.set_aspect('equal')
            ax.set_title("{}-{} (m)".format(hodo_list[i].lowerAlt, hodo_list[i].upperAlt), fontsize=9 )
            #if i==0:
            ax.legend(prop={'size':9})
            
            i += 1
            #plt.xlim([-5, 5])
            #plt.ylim([-5,5])
        
        plt.subplots_adjust(top=0.9, hspace=.6) 
        #plt.tight_layout(pad=1, w_pad=.5, h_pad=.5)   
        plt.tight_layout()
        
        plt.show() 
        """
    ########## FUNCTIONS ########## plotting courtesy of Kathyrin Reece
    def generatePoints(a,b,theta):
        print("  ")
        return none

    def hodoPlot(graph, x, y, low, high, index):
        graph.plot(x, y, color='black')
        points = np.array([x,y])

        # plot best fit ellipse
        xc = hodo_list[index].c_x
        yc = hodo_list[index].c_y
        a = hodo_list[index].a
        b = hodo_list[index].b
        theta = hodo_list[index].phi
        param = np.linspace(0, 2 * np.pi)    
        x = a * np.cos(param) * np.cos(theta) - b * np.sin(param) * np.sin(theta) + xc
        y = a * np.cos(param) * np.sin(theta) + b * np.sin(param) * np.cos(theta) + yc
        graph.plot(x, y, color='red')

        # plot rotated data
        # graph.plot(hodo_list[index].uRot, hodo_list[index].vRot)


        graph.arrow(xc, yc, a*np.cos(theta), a*np.sin(theta), width=0.03, head_length=0.15, length_includes_head=True, color='black')
        graph.set_aspect('equal')
        graph.set_title(str(low)+" - "+str(high))
        # graph.set_xlabel('u')
        # graph.set_ylabel('v')

        #print("hodoPlot: ell params: x,y,a,b,theta: ", xc, yc, a, b, theta)

    def trim(axs, n, rem):
        axs = axs.flat
        for ax in axs[n:]:
            ax.remove()
            rem += 1
        return axs[:n], rem

    ########## VARIABLES ##########
    global hodo
    hodo = []
    hodo = hodo_list
    index = 0
    rem = 0

    ########## MAIN ##########
    
    #Create List of Hodograph Files
    print("Plotting Local Hodographs")
    # for file in os.listdir(folder):
    #     hodo.append(file)

    #Create Plot Diagram
    num = len(hodo)
    cols = 4
    rows = np.ceil(num/cols)
    print("ROWS ", rows)
    #fig = plt.figure(figsize = (8.5, 2*rows), constrained_layout=True)
    fig = plt.figure(figsize = (8.5, 11), constrained_layout=True)
    axs = fig.subplots(int(rows),cols)#, sharex=True, sharey=True)
    fig.suptitle('T36 Local Hodographs')
    fig.text(.5, -.02, 'u wind speed')
    fig.text(-.02, .5, 'v wind speed', va='center', rotation='vertical')
    axs, rem = trim(axs, num, rem)
    
    #Fill Subplots
    for place in axs.flat:
            hodoPlot(place, hodo_list[index].u, hodo_list[index].v, hodo_list[index].lowerAlt, hodo_list[index].upperAlt, index)
            if rem >= 1 and index == (rows-2)*cols + cols-1:
                place.xaxis.set_tick_params(which='both', labelbottom=True)
            if rem >= 2 and index == (rows-2)*cols + cols-2:
                place.xaxis.set_tick_params(which='both', labelbottom=True)
            if rem >= 3 and index == (rows-2)*cols + cols-3:
                place.xaxis.set_tick_params(which='both', labelbottom=True)
            index += 1
    plt.show()

    return

def macroHodo():
    """ plot hodograph for entire flight
    """
    #plot v vs. u
    plt.figure("Macroscopic Hodograph", figsize=(10,10))  #Plot macroscopic hodograph
    plt.suptitle("Macro Hodograph for Entire Flight \n Background Wind Removed")
    c=Alt
    plt.scatter( u, v, c=c, cmap = 'magma', s = 1, edgecolors=None, alpha=1)
    #plt.plot(u,v)
    cbar = plt.colorbar()
    cbar.set_label('Altitude')  
    return

'''Interesting book recomendation: The tipping point by Malcolm Gladwell // Michael Pollan [coffee?]; per Dave Brown's reccommendation...'''

def uvVisualize():
    """ show u, v, background wind vs. altitude
    """
    #housekeeping
    plt.figure("U & V vs Time", figsize=(10,10)) 
    plt.suptitle('Smoothed U & V Components', fontsize=16)

    #u vs alt
    plt.subplot(1,2,1)
    plt.plot((u.magnitude + uBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(uBackground.magnitude, Alt.magnitude, label='Background - S.G.')
    plt.plot(u.magnitude, Alt.magnitude, label='De-Trended - S.G.')

    #plt.plot(uBackgroundRolling, Alt.magnitude, label='Background - R.M.')
    #plt.plot(uRolling, Alt.magnitude, label='De-Trended - R.M.')

    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.title("U")

    #v vs alt
    plt.subplot(1,2,2)
    plt.plot((v.magnitude + vBackground.magnitude), Alt.magnitude, label='Raw')
    plt.plot(vBackground.magnitude, Alt.magnitude, label='Background - S.G.')
    plt.plot(v.magnitude, Alt.magnitude, label='De-Trended - S.G.')

    #plt.plot(vBackgroundRolling, Alt.magnitude, label='Background - R.M.')
    #plt.plot(vRolling, Alt.magnitude, label='De-Trended - R.M.')

    plt.xlabel('(m/s)', fontsize=12)
    plt.ylabel('(m)', fontsize=12)
    plt.legend(loc='upper right', fontsize=16)
    plt.title("V")
    return



def manualTKGUI():
    """ tool for visualizing hodograph, sifting for micro hodographs, saving files for future analysis
    """
    
    class App:
        def __init__(self, master):
            """ method sets up gui
            """
            
            alt0 = 0
            wind0 = 100
            # Create a container
            tkinter.Frame(master)

            #CREATE ORIENTATION SELECTION
            
            self.orient = StringVar()
            self.orient.set(['CW', 'CCW'])
            self.picDir = tkinter.Spinbox(root, textvariable=self.orient, values=['CW', 'CCW'], state='readonly', wrap=True, font=Font(family='Helvetica', size=18, weight='normal'))
            self.picDir.place(relx=.05, rely=.5, relheight=.05, relwidth=.15)
            
            #END ORIENTATION SELECTION
            
            
            #Create Sliders
            self.low = IntVar()
            self.win = IntVar()
            self.up = IntVar()
            global winMin #minimum window size. edit here
            winMin = 15

            #create sliders
            self.lowSlider = tkinter.Scale(root, from_=0, to_=len(Alt.magnitude)-1, repeatinterval=1, orient=HORIZONTAL, command=self.updateSlideLow)
            self.lowSlider.place(relx=.05, rely=.15, relwidth=.15)
            self.upSlider = tkinter.Scale(root, from_=0, to_=len(Alt.magnitude)-1, repeatinterval=1, orient=HORIZONTAL, command=self.updateSlideUp)
            self.upSlider.place(relx=.05, rely=.35, relwidth=.15)
            self.winSlider = tkinter.Scale(root, from_=0, to_=len(Alt.magnitude)-1, repeatinterval=1, orient=HORIZONTAL, command=self.updateSlideWin)
            self.winSlider.place(relx=.05, rely=.25, relwidth=.15)
            

            #Create spinners
            self.lowSpinner = tkinter.Spinbox(root, command=self.updateLow, values=Alt.magnitude.tolist(), repeatinterval=1, font=Font(family='Helvetica', size=25, weight='normal'))
            self.lowSpinner.place(relx=.05, rely=.12, relheight=.05, relwidth=.15)  #originally followed above line
            self.upSpinner = tkinter.Spinbox(root, command=self.updateUp, values=Alt.magnitude.tolist(), repeatinterval=1,font=Font(family='Helvetica', size=25, weight='normal'))
            self.upSpinner.place(relx=.05, rely=.32, relheight=.05, relwidth=.15)
            self.winSpinner = tkinter.Spinbox(root, command=self.updateWin, from_=winMin, to=10000, repeatinterval=1, font=Font(family='Helvetica', size=25, weight='normal'))
            self.winSpinner.place(relx=.05, rely=.22, relheight=.05, relwidth=.15)  #originally followed above line

            #initialize vals inside spinners
            self.upSpinner.delete(0, 'end')
            self.upSpinner.insert(0, Alt.magnitude[15])

            #create lock radiobuttons
            global var #
            var = IntVar()
            var.set(1)
            self.lockLow = tkinter.Radiobutton(root, variable=var, value=1).place(relx=.03, rely=.14)
            self.lockUp = tkinter.Radiobutton(root, variable=var, value=2).place(relx=.03, rely=.24)
            self.lockWin = tkinter.Radiobutton(root, variable=var, value=3).place(relx=.03, rely=.34)

            #create labels
            self.lowLabel = tkinter.Label(root, text="Select Lower Altitude (m):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.09)
            self.upLabel = tkinter.Label(root, text="Select Upper Altitude (m):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.29)
            self.winLabel = tkinter.Label(root, text="Select Alt. Window (# data points):", font=Font(family='Helvetica', size=18, weight='normal')).place(relx=.05, rely=.19)

            #Create figure, plot 
            fig = Figure(figsize=(5, 4), dpi=100)
            self.ax = fig.add_subplot(111)
            fig.suptitle("{}".format(fileToBeInspected))
            self.l, = self.ax.plot(u[:15], v[:15], 'o', ls='-', markevery=[0])
            self.ax.set_aspect('equal')
        
            self.canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().place(relx=0.25, rely=0.05, relheight=.9, relwidth=.7)
            #frame.pack()
            
            #create buttons
            self.winLabel = tkinter.Label(root, text="Blue dot indicates lower altitude", font=Font(family='Helvetica', size=15, weight='normal')).place(relx=.05, rely=.4)
            self.quitButton = tkinter.Button(master=root, text="Quit", command=self._quit).place(relx=.05, rely=.7, relheight=.05, relwidth=.15)
            self.saveButton = tkinter.Button(master=root, text="Save Micro-Hodograph", command=self.save).place(relx=.05, rely=.6, relheight=.05, relwidth=.15)

            self.readyToSave = False #flag to make sure hodo is updated before saving
            #---------
            
        def updateSlideLow(self, *args):
            SlideLow = int(float(self.lowSlider.get()))
            self.lowSpinner.delete(0, 'end')
            self.lowSpinner.insert(0,Alt.magnitude[SlideLow])
            self.updateLow()
            return

        def updateSlideUp(self, *args):
            SlideUp = int(float(self.upSlider.get()))
            self.upSpinner.delete(0, 'end')
            self.upSpinner.insert(0,Alt.magnitude[SlideUp])
            self.updateUp()
            return

        def updateSlideWin(self, *args):
            SlideWin = int(float(self.winSlider.get()))
            self.winSpinner.delete(0, 'end')
            self.winSpinner.insert(0,SlideWin)
            self.updateWin()
            return

        def updateLow(self, *args):
            """ on each change to gui, this method refreshes hodograph plot
            """
            self.readyToSave = True


            valLow = int(float(self.lowSpinner.get()))
            valUp = int(float(self.upSpinner.get()))
            spinLow = np.where(Alt.magnitude == valLow)[0][0]
            spinUp = np.where(Alt.magnitude == valUp)[0][0]
            spinWin = int(self.winSpinner.get())
            lock = var.get()

            if lock == 1: #lower is locked, cancel
                print('error')
                spinLow = spinUp - spinWin
                self.lowSpinner.delete(0, 'end')
                self.lowSpinner.insert(0,Alt.magnitude[spinLow])
            if lock == 2: #window is locked, edit upper
                spinUp = spinLow + spinWin
                if spinUp >= len(Alt.magnitude): #if upper is above max, cancel
                    spinUp = len(Alt.magnitude)-1
                    spinLow = spinUp - spinWin
                    self.lowSpinner.delete(0, 'end')
                    self.lowSpinner.insert(0,Alt.magnitude[spinLow])
                else:
                    self.upSpinner.delete(0, 'end')
                    self.upSpinner.insert(0,Alt.magnitude[spinUp])
            if lock == 3: #upper is locked, edit window
                spinWin = spinUp-spinLow
                if spinWin < winMin: #if window is too small, cancel
                    spinWin = winMin
                    spinLow = spinUp - spinWin
                    self.lowSpinner.delete(0, 'end')
                    self.lowSpinner.insert(0,Alt.magnitude[spinLow])
                else:
                    self.winSpinner.delete(0, 'end')
                    self.winSpinner.insert(0,spinWin)

            self.lowSlider.set(spinLow)
            self.upSlider.set(spinUp)
            self.winSlider.set(spinWin)
            #get updated values from spinners
            low = np.where(Alt.magnitude == valLow)[0][0] #Gives index of current spinner altitude ##current altitude is sliderAlt
            up = np.where(Alt.magnitude == valUp)[0][0]
            win = spinWin

            #update graph to show between lower and upper variables
            self.l.set_xdata(u[low:up])
            self.l.set_ydata(v[low:up])

            self.ax.autoscale(enable=True)
            self.ax.relim()
            self.canvas.draw()
            return

        def updateUp(self, *args):
            """ on each change to gui, this method refreshes hodograph plot
            """
            self.readyToSave = True

            valLow = int(float(self.lowSpinner.get()))
            valUp = int(float(self.upSpinner.get()))
            spinLow = np.where(Alt.magnitude == valLow)[0][0]
            spinUp = np.where(Alt.magnitude == valUp)[0][0]
            spinWin = int(self.winSpinner.get())
            lock = var.get()

            if lock == 1: #lower is locked, edit window
                spinWin = spinUp-spinLow
                if spinWin < winMin: #if window is too small, cancel
                    spinWin = winMin
                    spinUp = spinLow + spinWin
                    self.upSpinner.delete(0, 'end')
                    self.upSpinner.insert(0,Alt.magnitude[spinUp])
                else:
                    self.winSpinner.delete(0, 'end')
                    self.winSpinner.insert(0,spinWin)
            if lock == 2: #window is locked, edit lower
                spinLow = spinUp - spinWin
                if spinLow < 0: #if lower is below 0, cancel
                    spinLow = 0
                    spinUp = spinLow + spinWin
                    self.upSpinner.delete(0, 'end')
                    self.upSpinner.insert(0,Alt.magnitude[spinUp])
                else:
                    self.lowSpinner.delete(0, 'end')
                    self.lowSpinner.insert(0,Alt.magnitude[spinLow])
            if lock == 3: #upper is locked, cancel
                spinUp = spinLow + spinWin
                self.upSpinner.delete(0, 'end')
                self.upSpinner.insert(0,Alt.magnitude[spinUp])

            self.lowSlider.set(spinLow)
            self.upSlider.set(spinUp)
            self.winSlider.set(spinWin)
            #get updated values from spinners
            low = np.where(Alt.magnitude == valLow)[0][0] #Gives index of current spinner altitude ##current altitude is sliderAlt
            up = np.where(Alt.magnitude == valUp)[0][0]
            win = spinWin

            #update graph to show between lower and upper variables
            self.l.set_xdata(u[low:up])
            self.l.set_ydata(v[low:up])

            self.ax.autoscale(enable=True)
            self.ax.relim()
            self.canvas.draw()
            return

        def updateWin(self, *args):
            """ on each change to gui, this method refreshes hodograph plot
            """
            self.readyToSave = True

            valLow = int(float(self.lowSpinner.get()))
            valUp = int(float(self.upSpinner.get()))
            spinLow = np.where(Alt.magnitude == valLow)[0][0]
            spinUp = np.where(Alt.magnitude == valUp)[0][0]
            spinWin = int(self.winSpinner.get())
            lock = var.get()

            if lock == 1: #lower is locked, edit upper
                spinUp = spinLow + spinWin
                if spinUp >= len(Alt.magnitude): #if upper is above max, cancel
                    spinUp = len(Alt.magnitude)-1
                    spinWin = spinUp - spinLow
                    self.winSpinner.delete(0, 'end')
                    self.winSpinner.insert(0,spinWin)
                else:
                    self.upSpinner.delete(0, 'end')
                    self.upSpinner.insert(0,Alt.magnitude[spinUp])
            if lock == 2: #window is locked, cancel
                spinWin = spinUp-spinLow
                self.winSpinner.delete(0, 'end')
                self.winSpinner.insert(0,spinWin)
            if lock == 3: #upper is locked, edit lower
                spinLow = spinUp - spinWin
                if spinLow < 0: #if lower is below 0, cancel
                    spinLow = 0
                    spinWin = spinUp - spinLow
                    self.winSpinner.delete(0, 'end')
                    self.winSpinner.insert(0,spinWin)
                else:
                    self.lowSpinner.delete(0, 'end')
                    self.lowSpinner.insert(0,Alt.magnitude[spinLow])

            self.lowSlider.set(spinLow)
            self.upSlider.set(spinUp)
            self.winSlider.set(spinWin)
            #get updated values from spinners
            low = np.where(Alt.magnitude == valLow)[0][0] #Gives index of current spinner altitude ##current altitude is sliderAlt
            up = np.where(Alt.magnitude == valUp)[0][0]
            win = spinWin

            #update graph to show between lower and upper variables
            self.l.set_xdata(u[low:up])
            self.l.set_ydata(v[low:up])
           
            self.ax.autoscale(enable=True)
            self.ax.relim()
            self.canvas.draw()
            return
        
        def save(self): 
            """ save current visible hodograph to .csv for further analysis
            """
            if self.readyToSave:
                
                ORIENTATION = self.orient.get()
                print('Orientation: ', ORIENTATION )
                spinLow = int(float(self.lowSpinner.get()))
                spinUp = int(float(self.upSpinner.get()))
                spinWindow = int(float(self.winSpinner.get()))
                lowerAltInd = np.where(Alt.magnitude == spinLow)[0][0]
                upperAltInd = lowerAltInd + spinWindow
            
            
                #collect local data for altitude that is visible in plot, dump into .csv
                ALT = Alt[lowerAltInd : upperAltInd]
                U = u[lowerAltInd : upperAltInd]
                V = v[lowerAltInd : upperAltInd]
                TEMP = Temp[lowerAltInd : upperAltInd]
                BV2 = bv2[lowerAltInd : upperAltInd]
                LAT = Lat[lowerAltInd : upperAltInd]
                LONG = Long[lowerAltInd : upperAltInd]
                TIME = Time[lowerAltInd : upperAltInd]
                
                instance = microHodo(ALT, U, V, TEMP, BV2, LAT, LONG, TIME, ORIENTATION)
                #instance.addOrientation(ORIENTATION)
                instance.addNameAddPath(fileToBeInspected, microHodoDir)
                instance.saveMicroHodoNoIndices()
                
            else: 
                print("Please Update Hodograph")
                    
            
            return

        def _quit(self):
            """ terminate gui
            """
            root.quit()     # stops mainloop
            root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
    root = tkinter.Tk()
    root.geometry("400x650")
    #Change the icon of window taskbar
    os.chdir(scriptDirectory)
    # from PIL import Image
    # filename = r'eclipse.png'
    # img = Image.open(filename)
    # img.save('logo.ico', sizes=[(255, 255)])
    # root.wm_iconbitmap("logo.ico")

    root.wm_title("Manual Hodograph GUI")
    App(root)
    root.mainloop()
    return

def run_(file, filePath):
    """ call neccesary functions 
    """
    #make sure there are no existing figures open
    plt.close('all')

    # set location of flight data as current working directory
    os.chdir(filePath)
    data = preprocessDataResample(file, flightData, spatialResolution, lowcut, highcut, order)
    
    
        
    if siftThruHodo:
       manualTKGUI()
       
    if analyze:
        hodo_list= doAnalysis(microHodoDir)
        
        print("hodo_list attributes: ", vars(hodo_list[0]).keys())

        if showVisualizations:

            #macroHodo()
            #uvVisualize()
            plotBulkMicros(hodo_list, file)
        return
    return
     
#Calls run_ method  
run_(fileToBeInspected, flightData) 
    
#last line