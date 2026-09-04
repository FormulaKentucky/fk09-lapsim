import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

def track_model(file_name, curvature_threshold):
    def interp(xy, L, dL, window):
        x = np.append(xy.loc[:,"x"].values, xy.loc[0,"x"])
        y = np.append(xy.loc[:,"y"].values, xy.loc[0,"y"])
        L = np.append(L, L[-1] + np.sqrt((x[-1]-x[-2])**2 + (y[-1]-y[-2])**2))
        
        xs = CubicSpline(L, x, bc_type='periodic')
        ys = CubicSpline(L, y, bc_type='periodic')
        
        xs_first = xs.derivative()
        ys_first = ys.derivative()
        
        L_fine = np.linspace(L[0], L[-1], int(1e+4))
        S_fine = cumulative_trapezoid(np.sqrt(xs_first(L_fine)**2 + ys_first(L_fine)**2), L_fine)
        S_fine = np.insert(S_fine, 0, 0)
        Ls = CubicSpline(S_fine, L_fine, bc_type='natural')
        
        S_new = np.arange(0, S_fine[-1], dL)
        L_new = Ls(S_new)
        
        xs_second = xs_first.derivative()
        ys_second = ys_first.derivative()
        
        dx_dL = xs_first(L_new)
        dy_dL = ys_first(L_new)
        d2x_dL2 = xs_second(L_new)
        d2y_dL2 = ys_second(L_new)
        
        curvature = (dx_dL * d2y_dL2 - dy_dL * d2x_dL2) / (dx_dL**2 + dy_dL**2)**(3/2)
        radii = 1 / np.abs(curvature)
        
        if window > 1:
            x_new = pd.Series(xs(L_new)).rolling(window=window, min_periods=1).mean().values
            y_new = pd.Series(ys(L_new)).rolling(window=window, min_periods=1).mean().values
        else:
            x_new = xs(L_new)
            y_new = ys(L_new)
        
        xy_new = pd.DataFrame({"x": x_new, "y": y_new})
        return xy_new, L_new, radii, curvature
    
    def deinterpolate(xy_new, L_new, L_original):
        xs = CubicSpline(L_new, xy_new.loc[:,"x"].values)
        ys = CubicSpline(L_new, xy_new.loc[:,"y"].values)
        x_orig = xs(L_original)
        y_orig = ys(L_original)
        return pd.DataFrame({"x": x_orig, "y": y_orig})
    
    def residual(params, xy_original, L_original):
        dL, window = params
        xy_new, L_new, _, curvature = interp(xy_original, L_original, dL, max(1, int(window)))
        
        xy_deinterp = deinterpolate(xy_new, L_new, L_original)
        res = np.sqrt((xy_deinterp["x"].values - xy_original["x"].values)**2 + 
                      (xy_deinterp["y"].values - xy_original["y"].values)**2)
        
        curvature_fft = np.fft.fft(curvature)
        freq_amplitude = np.abs(curvature_fft)
        
        low_amp_mask = freq_amplitude < curvature_threshold
        penalty = np.zeros_like(freq_amplitude)
        penalty[low_amp_mask] = 1e+16 / np.exp(freq_amplitude[low_amp_mask])
        
        penalty_time = np.real(np.fft.ifft(penalty))
        res = res + penalty_time[:len(res)]
        
        return res
    
    track_raw = pd.read_csv(file_name, sep="\t")
    xy_OG = track_raw.loc[:,("x","y")]
    L = track_raw.loc[:,"dist"].values
    
    result = least_squares(residual, [0.5, 30.0], args=(xy_OG, L), 
                           bounds=([0.05, 1.0], [2.0, 100.0]))
    
    dL_opt, window_opt = result.x
    xy_new, L_new, radii, _ = interp(xy_OG, L, dL_opt, max(1, int(window_opt)))
    xy_final, L_final, radii_final, _ = interp(xy_new, L_new, 0.01, 1)
    
    return xy_final, L_final, radii_final, xy_OG
    
    
    
