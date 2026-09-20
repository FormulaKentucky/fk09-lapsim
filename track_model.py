import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares


def track_model(file_name, curvature_threshold):
    def interp(xy, L, dL, window):
        x = np.append(xy["x"].values, xy["x"].values[0])
        y = np.append(xy["y"].values, xy["y"].values[0])
        L = np.append(L, L[-1] + np.hypot(x[-1] - x[-2], y[-1] - y[-2]))

        xs = CubicSpline(L, x, bc_type='periodic')
        ys = CubicSpline(L, y, bc_type='periodic')
        xs_d = xs.derivative()
        ys_d = ys.derivative()

        # Fine grid must scale with track length / dL, not be fixed at 1e4
        n_fine = max(int(1e4), int((L[-1] - L[0]) / dL) * 20)
        L_fine = np.linspace(L[0], L[-1], n_fine)
        S_fine = np.concatenate(
            [[0.0],
             cumulative_trapezoid(np.hypot(xs_d(L_fine), ys_d(L_fine)), L_fine)]
        )
        total = S_fine[-1]
        Ls = CubicSpline(S_fine, L_fine, bc_type='natural')

        # arange guarantees spacing == dL exactly in S
        S_new = np.arange(0.0, total, dL)
        L_new = Ls(S_new)

        xs_dd = xs_d.derivative()
        ys_dd = ys_d.derivative()
        dx_dL = xs_d(L_new)
        dy_dL = ys_d(L_new)
        d2x_dL2 = xs_dd(L_new)
        d2y_dL2 = ys_dd(L_new)

        curvature = (dx_dL * d2y_dL2 - dy_dL * d2x_dL2) / (dx_dL**2 + dy_dL**2) ** 1.5
        radii = 1.0 / np.abs(curvature)

        x_new = xs(L_new)
        y_new = ys(L_new)
        if window > 1:
            x_new = pd.Series(x_new).rolling(window=window, min_periods=1).mean().values
            y_new = pd.Series(y_new).rolling(window=window, min_periods=1).mean().values

        xy_new = pd.DataFrame({"x": x_new, "y": y_new})
        # Return S_new (true arclengths), NOT L_new (parameter values)
        return xy_new, S_new, radii, curvature

    def deinterpolate(xy_new, L_new, L_original):
        xs = CubicSpline(L_new, xy_new["x"].values)
        ys = CubicSpline(L_new, xy_new["y"].values)
        return pd.DataFrame({"x": xs(L_original), "y": ys(L_original)})

    def residual(params, xy_original, L_original):
        dL, window = params
        xy_new, L_new, _, curvature = interp(xy_original, L_original, dL, max(1, int(window)))

        xy_deinterp = deinterpolate(xy_new, L_new, L_original)
        res = np.hypot(xy_deinterp["x"].values - xy_original["x"].values,
                       xy_deinterp["y"].values - xy_original["y"].values)

        amp = np.abs(np.fft.fft(curvature))
        penalty = np.zeros_like(amp)
        mask = amp < curvature_threshold
        penalty[mask] = 1e16 / np.exp(amp[mask])
        res = res + np.real(np.fft.ifft(penalty))[:len(res)]
        return res

    track_raw = pd.read_csv(file_name, sep="\t")
    xy_OG = track_raw.loc[:, ("x", "y")]
    L = track_raw.loc[:, "dist"].values

    result = least_squares(residual, [0.5, 30.0], args=(xy_OG, L),
                           bounds=([0.05, 1.0], [2.0, 100.0]))

    dL_opt, window_opt = result.x
    xy_smooth, L_smooth, _, _ = interp(xy_OG, L, dL_opt, max(1, int(window_opt)))
    xy_final, L_final, radii_final, _ = interp(xy_smooth, L_smooth, 0.1, 0)

    return xy_final, L_final, L, radii_final, xy_OG
    
    
    
