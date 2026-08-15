import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

g = 9.80665 # Graviational acceleration [m / s²]

vehicle_parameters = pd.read_json("data/parameters.json")

def powertrain_model(vp, dyno_file):
    # Read in the dyno curves
    dyno_curves = pd.read_csv(dyno_file)
    RPM = dyno_curves.loc[:,'RPM'].values 
    torque = dyno_curves.loc[:,'Torque [ft-lb]'].values * 1.3558 # [N-m]
    power = dyno_curves.loc[:,'Power [hp]'].values * 0.7457 # [kW]
    
    # Read in the vehicle parameters
    final_drive = vp.loc['Final Drive Ratio','Powertrain']
    primary_gear = vp.loc['Primary Gear Ratio','Powertrain']
    total_gears = np.zeros(6)
    for i in range(1,7):
        total_gears[i-1] = vp.loc[str(i) + ' Gear Ratio','Powertrain'] * final_drive * primary_gear
    
    # Find angular velocity and torque of the wheels
    omega_wheel = np.tile(RPM, (6, 1)) / np.tile(total_gears, (len(RPM), 1)).T # [RPM]
    tau_wheel = np.tile(torque, (6, 1)) * np.tile(total_gears, (len(RPM), 1)).T 
    
    # Find the total speed of the car
    V_car = omega_wheel * vp.loc['Tire Rolling Radius','Tires'] * 2 * np.pi / 60 # [m / s]
    
    # Find the tractive force 
    F_trac = tau_wheel / vp.loc['Tire Rolling Radius','Tires'] # [N]
    
    # Create a new velocity grid
    V_opt = np.linspace(0, np.max(V_car), 10000)   
    F_trac_total = np.zeros((6, len(V_opt)))   
    
    # Organize the tractive forces by velocity
    for gear in range(6):
        v_row = V_car[gear]
        f_row = F_trac[gear]
        # np.interp extrapolates using edge values; we force zero outside
        mask = (V_opt >= v_row.min()) & (V_opt <= v_row.max())
        F_trac_total[gear, mask] = np.interp(V_opt[mask], v_row, f_row)
        
    # Find the optimal tractive force and the velocity it occurs at
    F_opt = np.zeros(len(V_opt))
    gear_opt = np.zeros(len(V_opt))
    for i in range(len(V_opt)):
        F_opt[i] = np.max(F_trac_total[:,i])
        gear_opt[i] = np.argmax(F_trac_total[:,i]) + 1 # Optimal gear
        
        if V_opt[i] <= np.min(V_car):
            F_opt[i] = F_trac[0,0]

    # Initialize output array
    RPM_opt = np.zeros(10000)
    
    # Minimum possible speed and RPM
    V_min = np.min(V_car)          # smallest speed in any gear at min RPM
    RPM_min = np.min(RPM)
    
    # 1. Speeds that cannot be reached (e.g., 0 → V_min) → idle RPM
    low_mask = V_opt <= V_min
    RPM_opt[low_mask] = RPM_min
    
    # 2. For each gear, interpolate the RPM that gives V_opt
    for gear in range(1, 7):
        mask = (gear_opt == gear) & ~low_mask
        if not np.any(mask):
            continue
        # Speed‑RPM curve for this gear
        speed_curve = V_car[gear - 1, :]   # strictly increasing with RPM
        RPM_opt[mask] = np.interp(V_opt[mask], speed_curve, RPM)
    
    
    return [RPM_opt, F_opt, V_opt, gear_opt, RPM, V_car, F_trac]

def force_model(vp, WF, pt, V_car):
    # Define the tire_model as a force subsidiary
    def tire_model(vp, F_zf, F_zr):
        # Bring in the tire parameters
        mu_x = vp.loc['Longitudinal Friction Coefficient','Tires']
        mu_y = vp.loc['Lateral Friction Coefficient','Tires']
        mu_x_norm = vp.loc['Longitudinal Friction Normal Load Rating','Tires']
        mu_y_norm = vp.loc['Lateral Friction Normal Load Rating','Tires']
        mu_x_sens = vp.loc['Longitudinal Friction Sensitivity','Tires']
        mu_y_sens = vp.loc['Lateral Friction Sensitivity','Tires']

        # Compute front and rear sensitivity corrected longitudinal friction coefficients [-]
        mu_xf = mu_x + mu_x_sens * (mu_x_norm * g - 0.5 * F_zf)
        mu_xr = mu_x + mu_x_sens * (mu_x_norm * g - 0.5 * F_zr)
        
        # Compute front and rear sensitivity corrected lateral friction coefficients [-]
        mu_yf = mu_y + mu_y_sens * (mu_y_norm * g - 0.5 * F_zf)
        mu_yr = mu_y + mu_y_sens * (mu_y_norm * g - 0.5 * F_zr)
        
        return [mu_xf, mu_xr, mu_yf, mu_yr]
    
    
    """
    ---------------------------------------------------------------------------
                        VERTICAL FORCE CALCULATIONS
    ---------------------------------------------------------------------------
    """
    

    # Determine front and rear vehicle weight [N]
    W_f = vp.loc["Mass","General"] * g * vp.loc["Front Weight Distribution","General"] / 100 - WF * g 
    W_r = vp.loc["Mass","General"] * g * (100 - vp.loc["Front Weight Distribution","General"]) / 100 + WF * g 
    
    # Determine fornt and rear downforce coefficients [-]
    C_zf = vp.loc["Total Downforce Coefficient","Aero"] * vp.loc["Front Aero Balance","Aero"] / 100
    C_zr = vp.loc["Total Downforce Coefficient","Aero"] * (100 - vp.loc["Front Aero Balance","Aero"]) / 100
    
    # Determine the front and rear aerodynamic downforces [N]
    F_Czf = 0.5 * vp.loc["Air Density","Aero"] * C_zf * vp.loc["Frontal Area","Aero"] * V_car ** 2
    F_Czr = 0.5 * vp.loc["Air Density","Aero"] * C_zr * vp.loc["Frontal Area","Aero"] * V_car ** 2
    
    # Determine the front, rear, and total vertical load on the car [N]
    F_zf = W_f + F_Czf
    F_zr = W_r + F_Czr
    F_z = F_zf + F_zr 



    """
    ---------------------------------------------------------------------------
                       LONGITUDINAL FORCE CALCULATIONS
    ---------------------------------------------------------------------------
    """
    
    
    
    # Determine the drag force on the car [N]
    F_Cx = 0.5 * vp.loc["Air Density","Aero"] * vp.loc["Total Drag Coefficient","Aero"] * vp.loc["Frontal Area","Aero"] * V_car ** 2
    
    # Determine the rolling resistance of the car [N]
    F_Crx = vp.loc["Rolling Resistance Coefficient","Tires"] * F_z
    
    mu = tire_model(vp,F_zf,F_zr)
    
    # Determine the force allowed by tire traction RWD [N]
    F_tires_acc = np.abs(mu[1] * F_zr)
    
    # Determine the front, rear, and total tire deceleration [N]
    F_tires_decf = mu[0] * F_zf
    F_tires_decr = mu[1] * F_zr
    F_tires_dec = F_tires_decf + F_tires_decr
    
    # Determine the tractive force via interpolation
    F_trac_interp = interp1d(pt[2],pt[1],kind='linear')
    F_trac = F_trac_interp(V_car)
    
    # Final positive longitudinal force [N]
    F_x_acc = np.min([F_trac, F_tires_acc])
    
    # Final negative longitudinal force [N]
    F_x_dec = F_Cx + F_Crx
    
    # Determine the max longitudinal force possible [N]
    F_x = F_x_acc - F_x_dec if F_x_acc - F_x_dec > 0.0 else 0.0
    
    return [F_x_acc, F_x_dec, np.abs(F_x_dec)]
    
    
    
    
    
    
    
    
    