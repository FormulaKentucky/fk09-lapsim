import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from matplotlib import pyplot as plt

def track_model(file_name):
    # This function interpolates and smoothes the track data (if window > 1)
    def interp(xy, L, dL, window):
        # Split up the data and ensure the first and last positions are the same
        x = np.append(xy.loc[:,"x"].values, xy.loc[0,"x"])
        y = np.append(xy.loc[:,"y"].values, xy.loc[0,"y"])
        
        # Match the length array
        L = np.append(L, L[-1] + np.sqrt((x[-1]-x[-2])**2 + (y[-1]-y[-2])**2))
    
        
        # Make the spline functions
        xs = CubicSpline(L, x, bc_type='periodic')
        ys = CubicSpline(L, y, bc_type='periodic')
        
        # Get the first derivatives for each spline 
        xs_first = xs.derivative()
        ys_first = ys.derivative()
        
        # Define the parametric curve and interpolate L to its grid
        L_fine = np.linspace(L[0], L[-1], int(1e+4))
        S_fine = cumulative_trapezoid(np.sqrt(xs_first(L_fine)**2 + ys_first(L_fine)**2), L_fine)
        S_fine = np.insert(S_fine, 0, 0)
        Ls = CubicSpline(S_fine, L_fine, bc_type='natural')
        
        # Compute the new length grid
        S_new = np.arange(0, S_fine[-1], dL)
        L_new = Ls(S_new)
        
        # Compute the radii for the curve
        xs_second = xs_first.derivative()
        ys_second = ys_first.derivative()
        
        dx_dL = xs_first(L_new)
        dy_dL = ys_first(L_new)
        d2x_dL2 = xs_second(L_new)
        d2y_dL2 = ys_second(L_new)
        
        radii = (dx_dL**2 + dy_dL**2)**(3/2) / np.abs(dx_dL * d2y_dL2 - dy_dL * d2x_dL2)
        
        x_new = xs(L_new)
        y_new = ys(L_new)
        xy_new = pd.DataFrame([x_new,y_new]).T.rolling(window=window,min_periods=1).mean()
        xy_new.columns = ["x", "y"]
        return xy_new, L_new, radii
    
    # Read in the track data then split up x and y positions
    track_raw = pd.read_csv(file_name, sep="\t")
    xy = track_raw.loc[:,("x","y")]
    
    # Get the track length values
    L = track_raw.loc[:,"dist"].values
    
    # Do a couple of iterations on this (maybe find a way to best fit?)
    xy, L, _ = interp(xy, L, 0.5, 30)
    xy, L, _ = interp(xy, L, 0.2, 50)
    return interp(xy, L, 0.1, 1) # Final window should always be 1 for correct dL
    
    
    
xy, L, radii = track_model("data/11-22-25 AutoX.txt")
    
    
    
