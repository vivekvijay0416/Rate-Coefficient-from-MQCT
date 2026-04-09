import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# =================================================================
# 1. CONSTANTS & CONFIGURATION
# =================================================================
K_TO_WN = 0.6950345740        # Boltzmann constant (cm^-1/K)
RATE_CONSTANT = 4.84997439836149E-13  # Rate prefactor (cm^3/s)
EXTRAP_MIN_U = 40.0 

# --- CROSS SECTION METHOD ---
# "direct"  -> Uses sigmaU_qu directly.
# "derived" -> Uses sigmaU_ex and detailed balance: sigma_qu = sigma_ex * (2j_1 + 1)/(2j_2 + 1)
# "average" -> Averages the "direct" and "derived" methods.
CROSS_SECTION_METHOD = "average"  

# --- RATE CALCULATION METHOD ---
# "hybrid" -> Uses raw data for U < EXTRAP_MIN_U, and fitted curve for U >= EXTRAP_MIN_U
# "fitted" -> Uses the analytical fit/extrapolation for all U > U0
RATE_METHOD = "hybrid"
VIS_HYBRID  =  True

# --- EXTRAPOLATION PARAMETER 'A' ---
# "fixed"      -> Hardcodes A = 1 (Original 2-point fit)
# "manual"     -> Uses the value set in EXTRAP_A_MANUAL_VALUE
# "calculated" -> Computes A dynamically using the 3-point fit formula
EXTRAP_A_METHOD = "manual"  
EXTRAP_A_MANUAL_VALUE = 0.5    # Only used if EXTRAP_A_METHOD = "manual"


# --- TEMPERATURES & PLOTTING ---
TEMPS = [5, 10, 20, 30, 50, 80, 100, 150, 200, 300]
VISUALIZE_TEMPS = [10, 50, 100, 300] # Must be exactly 4 temperatures from TEMPS


# --- FILE PATHS ---
# NEW: Added path for the MQCT User Input file to map states properly
USER_INPUT_PATH = r"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/USER_INPUT_CHECK.out"
DB_FILE_PATH =  r"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/E_he_Database.dat"
OUTPUT_TXT_PATH =rf"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/Rate_Coefficients_full_range_hybrid_U_min_40.dat"
VIZ_SAVE_PATH = rf"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/plots_rate_co-eff/full_enr_range_large_A_{EXTRAP_A_MANUAL_VALUE}"

# =================================================================
# 2. STATE PARSING & MAPPING
# =================================================================

def parse_states(filepath):
    """
    Reads USER_INPUT_CHECK.out to map State Index -> Quantum Numbers.
    Returns: states[idx] = {'J': int, 'Ka': int, 'Kc': int, 'tau': int, 'E': float, 'qn': str}
    """
    states = {}
    print(f"Reading states from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_reading = False
    for line in lines:
        if "CHANNELS ENERGIES" in line:
            start_reading = True
            continue
        
        if start_reading:
            parts = line.split()
            # Expecting: Index  J  Ka  Kc  Energy
            if len(parts) >= 5 and parts[0].isdigit():
                try:
                    idx = int(parts[0])
                    J = int(parts[1])
                    Ka = int(parts[2])
                    Kc = int(parts[3])
                    E = float(parts[4])
                    
                    tau = Ka - Kc
                    qn_str = f"{J}_{Ka}_{Kc}" 
                    
                    states[idx] = {
                        'J': J, 'Ka': Ka, 'Kc': Kc, 
                        'tau': tau, 'E': E, 'qn': qn_str
                    }
                except ValueError:
                    continue
    return states

def map_db_to_user_input(df_raw, states_dict):
    """Uses the parsed states dictionary to map j_1, ka_1, kc_1 to exact MQCT indices."""
    # Create a reverse lookup dictionary: (J, Ka, Kc) -> Index
    rev_map = {}
    for idx, info in states_dict.items():
        key = (info['J'], info['Ka'], info['Kc'])
        rev_map[key] = idx
        
    # Apply mapping to database
    # State 1 (lower energy) = flv for quenching
    df_raw['flv'] = df_raw.apply(lambda r: rev_map.get((r['j_1'], r['ka_1'], r['kc_1'])), axis=1)
    
    # State 2 (higher energy) = ilv for quenching
    df_raw['ilv'] = df_raw.apply(lambda r: rev_map.get((r['j_2'], r['ka_2'], r['kc_2'])), axis=1)
    
    # Drop any rows that failed to map (missing states in user input) and cast to int
    df_mapped = df_raw.dropna(subset=['ilv', 'flv']).copy()
    df_mapped['ilv'] = df_mapped['ilv'].astype(int)
    df_mapped['flv'] = df_mapped['flv'].astype(int)
    
    return df_mapped

# =================================================================
# 3. MATH & EXTRAPOLATION LOGIC
# =================================================================

def handle_duplicates(u_arr, sigma_arr):
    df = pd.DataFrame({'u': u_arr, 's': sigma_arr})
    df = df.groupby('u')['s'].mean().reset_index().sort_values('u')
    return df['u'].values, df['s'].values

def get_sigma_func_with_bounds(U_raw, sigma_raw, U0):
    U_data, sigma_U = handle_duplicates(U_raw, sigma_raw)
    mask = sigma_U > 1e-35
    U_filt, S_filt = U_data[mask], sigma_U[mask]
    
    if len(U_filt) < 3: return (lambda u: 0.0), U0, U0

    valid_indices = np.where(U_filt >= EXTRAP_MIN_U)[0]
    start_idx = valid_indices[0] if len(valid_indices) >= 2 else 0

    u_boundary_low = U_filt[start_idx + 1] if start_idx + 1 < len(U_filt) else U_filt[-1]
    u_boundary_high = U_filt[-2]

    # Extract points for low-energy fit
    x1, yy1 = U_filt[start_idx], S_filt[start_idx]
    dx1 = x1 - U0
    
    # --- CALCULATE A, B, and C ---
    if start_idx + 2 < len(U_filt):
        x2, yy2 = U_filt[start_idx+1], S_filt[start_idx+1]
        x3, yy3 = U_filt[start_idx+2], S_filt[start_idx+2]
        dx2, dx3 = x2 - U0, x3 - U0
        
        if EXTRAP_A_METHOD == "fixed":
            A = 1.0
        elif EXTRAP_A_METHOD == "manual":
            A = float(EXTRAP_A_MANUAL_VALUE)
        elif EXTRAP_A_METHOD == "calculated":
            try:
                num = (x3 - x1)*np.log(yy2/yy1) - (x2 - x1)*np.log(yy3/yy1)
                den = (x3 - x1)*np.log(dx2/dx1) - (x2 - x1)*np.log(dx3/dx1)
                A = num / den
                # Safeguard: if A is invalid or negative, fall back to 1.0
                if np.isnan(A) or np.isinf(A) or A <= 0: A = 1.0
 
            except:
                A = 1.0
        else:
            A = 1.0

        try:
            B = (x2 - x1) / np.log( (yy1/yy2) * ((dx2/dx1)**A) )
            # Exact formula from the paper
            C = (dx1 / ((yy1/A)**(1.0/A))) * np.exp(-dx1 / (A*B))
        except:
            B = 5.0 * dx1
            C = (dx1 / ((yy1/A)**(1.0/A))) if yy1>0 else 1.0

    elif start_idx + 1 < len(U_filt):
        # Fallback to 2-point fit
        x2, yy2 = U_filt[start_idx+1], S_filt[start_idx+1]
        dx2 = x2 - U0
        A = float(EXTRAP_A_MANUAL_VALUE) if EXTRAP_A_METHOD == "manual" else 1.0
        try:
            B = (x2 - x1) / np.log( (yy1/yy2) * ((dx2/dx1)**A) )
            C = (dx1 / ((yy1/A)**(1.0/A))) * np.exp(-dx1 / (A*B))
        except:
            B = 5.0 * dx1
            C = (dx1 / ((yy1/A)**(1.0/A))) if yy1>0 else 1.0
    else:
        # Fallback to 1 point
        A = float(EXTRAP_A_MANUAL_VALUE) if EXTRAP_A_METHOD == "manual" else 1.0
        B = 5.0 * dx1
        C = (dx1 / ((yy1/A)**(1.0/A))) if yy1>0 else 1.0

    # --- REGIME 3: TAIL ---
    xn_1, xn = U_filt[-2], U_filt[-1]
    yn_1, yn = S_filt[-2], S_filt[-1]
    try:
        b_tail = -(xn_1 - xn) / np.log(yn_1 / yn)
        a_tail = yn * np.exp(xn / b_tail)
    except:
        a_tail, b_tail = yn, 500.0

    # --- REGIME 2: SPLINE ---
    U_spline, S_spline = U_filt[(start_idx+1):], S_filt[(start_idx+1):]
    kind = 'cubic' if len(U_spline) >= 4 else 'linear'
    interp = interp1d(np.log10(U_spline), np.log10(S_spline), kind=kind, fill_value='extrapolate') if len(U_spline) > 1 else lambda x: np.log10(S_filt[-1])

    # --- PIECEWISE FUNCTION ---
    def sigma_func(U):
        if U <= U0: 
            return 0.0
        elif U < u_boundary_low:
            # Safety net to prevent divide-by-zero crashes
            if C == 0.0 or np.isinf(C) or np.isnan(C): 
                return 0.0
            return max(0.0, A * ((U - U0) / C)**A * np.exp(-(U - U0) / B))
        elif U <= u_boundary_high:
            return 10**interp(np.log10(U))
        else:
            return a_tail * np.exp(-U / b_tail)
    
    
    return sigma_func, u_boundary_low, u_boundary_high
# =================================================================
# 4.1. Data Analysis : PLOTTING FUNCTION (2x3 GRID)
# =================================================================

def plot_detailed_transition(ilv, flv, dE_real, g_i, q_u, q_su, q_fn, q_low, q_high, U0, temps, viz_temps, rates_fitted, rates_hybrid, integrand_funcs_map):
    print(f"\nGenerating detailed plot for transition {ilv} -> {flv}...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # ==========================================
    # 1. Define U_min masks FIRST so we can color 
    #    the points in ax1 and the integrand plots
    # ==========================================
    mask_used = q_u >= EXTRAP_MIN_U
    u_used, s_used = q_u[mask_used], q_su[mask_used]
    u_ign, s_ign = q_u[~mask_used], q_su[~mask_used]

    # ==========================================
    # 2. ax1: Cross Section Plot
    # ==========================================
    ax1 = axes[0]
    # Plot a faint connecting line for visual continuity
    ax1.plot(q_u, q_su, '-', color='gray', alpha=0.4)
    # Scatter the split points with the new colors
    if len(u_ign) > 0:
        ax1.plot(u_ign, s_ign, color='darkred',linewidth=1, marker='o', alpha=0.6, label='< U min (Resonace)')
    ax1.plot(u_used, s_used, color='darkgreen',linewidth=1, marker='o', alpha=0.6, label='>= U min ')
    
    ax1.set_title(rf"Cross Section State {ilv} $\to$ State {flv}")
    ax1.set_xlabel(rf"U (cm$^{{-1}}$)", fontsize=12)
    ax1.set_ylabel(rf"$\sigma_U$ ({CROSS_SECTION_METHOD}) ($\AA^2$)", fontsize=12)
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # ==========================================
    # 3. ax3: Rate Coefficients Plot (vs Temp)
    # ==========================================
    ax3 = axes[3]
    ax3.plot(temps, rates_fitted, 's-', color='b', label='Analytic Extrap')
    if VIS_HYBRID:
        ax3.plot(temps, rates_hybrid, 'o--', color='darkred', label='Actual Data + Fit')
    ax3.set_title("Rate Coefficients")
    ax3.set_xlabel("T (K)", fontsize=12)
    ax3.set_ylabel(r"k (cm$^3$/s)", fontsize=12)
    ax3.legend()
    ax3.grid(True, which='both', alpha=0.3)

    # ==========================================
    # 4. Integrand Subplots
    # ==========================================
    plot_indices = [1, 2, 4, 5]

    def calc_raw_contribution(u_val, sigma_val, temp_k):
        kt_val = K_TO_WN * temp_k
        u_safe = max(u_val, 1e-9)
        term = (abs(dE_real)/(4*u_safe))**2
        prefactor = g_i * u_safe * (1.0 - term) * np.exp(-u_safe/kt_val * (1 + term))
        return prefactor * sigma_val

    for idx, t_val in zip(plot_indices, viz_temps):
        if t_val not in integrand_funcs_map: continue
        ax = axes[idx]
        func = integrand_funcs_map[t_val]
        
        u_grid = np.logspace(np.log10(max(0.1, U0)), 4, 1000)
        y_curve = [func(u) for u in u_grid]
        ax.plot(u_grid, y_curve, 'b-', linewidth=1.5, label='Integrand Fit')

        # 'Used' data -> Dark Green
        y_pts_used = [calc_raw_contribution(u, s, t_val) for u, s in zip(u_used, s_used)]
        ax.scatter(u_used, y_pts_used, color='darkgreen', marker='o', s=20, label='< U min ', zorder=5)

        # 'Ignored' data (before U_min) -> Dark Red
        if len(u_ign) > 0:
            y_pts_ign = [calc_raw_contribution(u, s, t_val) for u, s in zip(u_ign, s_ign)]
            ax.scatter(u_ign, y_pts_ign, color='darkred', marker='o', s=20, label='>= U min ', zorder=5)
            ax.plot(u_ign, y_pts_ign, linestyle=':', color='black', linewidth=1.5, alpha=0.7)

        ax.axvspan(U0, q_low, color='orange', alpha=0.15, label='Extrapolation')
        ax.axvspan(q_low, q_high, color='green', alpha=0.1, label='Spline')
        ax.axvspan(q_high, 10000, color='orange', alpha=0.15)
        ax.set_title(f"Integrand at T={t_val}K")
        ax.set_xlabel("U (cm$^{-1}$)")
        ax.set_ylabel("Contribution to Rate")
        ax.set_xscale('log')
        #ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        if idx == 1: ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    
    if VIZ_SAVE_PATH:
        os.makedirs(VIZ_SAVE_PATH,exist_ok=True)
       
        save_file = os.path.join(VIZ_SAVE_PATH, f"Transition_{ilv}_{flv}.png")
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"  -> Detailed plot saved to: {save_file}")
        plt.close(fig)
    else:
        plt.show()

# =================================================================
# 4.2. Data Analysis : Analysing Rate differnce between two methods
# =================================================================

def analyze_saved_rate_differences(hybrid_txt_path, fitted_txt_path, check_temps, threshold_pct=5.0, max_ilv=None, min_rate=1e-15, save_dir=None):
    """
    Reads two pre-calculated rate coefficient text files (Hybrid and Fitted),
    calculates the percentage difference, prints outliers, and plots diagonals.
    
    Parameters:
    - hybrid_txt_path: Path to the Rate_Coefficients_Hybrid.txt file
    - fitted_txt_path: Path to the Rate_Coefficients_Fitted.txt file
    - check_temps: List of temperatures to analyze (e.g., [10, 50, 300])
    - threshold_pct: The % difference above which a transition is printed to the console
    - max_ilv: Optional integer. If set, ignores any transitions originating from states higher than this index.
    - save_dir: Directory to save the plots. If None, plots are shown interactively.
    - min_rate: The absolute minimum rate coefficient (cm^3/s) worth analyzing. 
                Anything below this is considered negligible and ignored.
    """
    
    print("Reading and merging rate coefficient databases...")
    try:
        # Read the text files, skipping row 1 (the dashed line '--------')
        df_hyb = pd.read_csv(hybrid_txt_path, sep=r'\s+', skiprows=[1])
        df_fit = pd.read_csv(fitted_txt_path, sep=r'\s+', skiprows=[1])
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Merge the two databases on the state indices (ilv, flv)
    df_merged = pd.merge(df_hyb, df_fit, on=['ilv', 'flv'], suffixes=('_hyb', '_fit'))

    # --- Filter by Initial State (ilv) ---
    if max_ilv is not None:
        original_len = len(df_merged)
        df_merged = df_merged[df_merged['ilv'] <= max_ilv].copy()
        theoretical_total = int(max_ilv * (max_ilv - 1) / 2) # N(N-1)/2 for quenching
        print(f"Applied Initial State Threshold: ilv <= {max_ilv}")
        print(f"Filtered out {original_len - len(df_merged)} high-energy transitions.")
    else:
        theoretical_total = None

    print("\n" + "="*75)
    print(f" HYBRID vs FITTED RATE DIAGNOSTICS (> {threshold_pct}% Difference)")
    print(f" (Filtering out all rates < {min_rate:.1E} cm^3/s)")
    print("="*75)

    for T in check_temps:
        col_hyb = f"{T}K_hyb"
        col_fit = f"{T}K_fit"

        if col_hyb not in df_merged.columns or col_fit not in df_merged.columns:
            print(f"Warning: Temperature {T}K not found in one or both files.")
            continue

        # Force columns to numeric, turning weird strings into NaNs
        df_merged[col_hyb] = pd.to_numeric(df_merged[col_hyb], errors='coerce')
        df_merged[col_fit] = pd.to_numeric(df_merged[col_fit], errors='coerce')

        hyb_vals = df_merged[col_hyb].values
        fit_vals = df_merged[col_fit].values

        # --- THE FIX: np.isfinite() strictly blocks both NaNs AND Infinity ---
        valid_mask = (hyb_vals >= min_rate) & np.isfinite(hyb_vals) & np.isfinite(fit_vals)
        
        total_valid = np.sum(valid_mask)
        ignored_low_rate = len(hyb_vals) - total_valid
        
        pct_diff = np.zeros_like(hyb_vals)
        pct_diff[valid_mask] = (np.abs(hyb_vals[valid_mask] - fit_vals[valid_mask]) / hyb_vals[valid_mask]) * 100.0

        # Flag the outliers among the valid ones
        df_merged[f"{T}K_diff_%"] = pct_diff
        outliers = df_merged[(df_merged[f"{T}K_diff_%"] > threshold_pct) & valid_mask]
        outlier_count = len(outliers)
        
        outlier_pct_of_valid = (outlier_count / total_valid * 100.0) if total_valid > 0 else 0.0

        # --- Print Console Report ---
        print(f"\n--- Temperature: {T} K ---")
        
        if theoretical_total:
            print(f"  Total valid transitions analyzed : {total_valid} (Theoretical max: {theoretical_total})")
        else:
            print(f"  Total valid transitions analyzed : {total_valid}")
            
        print(f"  Transitions ignored (too small)  : {ignored_low_rate} (k < {min_rate:.1E} or NaN/Inf)")
        print(f"  Transitions > {threshold_pct}% difference  : {outlier_count} ({outlier_pct_of_valid:.1f}% of analyzed)")
        print(f"  ------------------------------------------------")
        
        if total_valid == 0:
            print(f"  No rate coefficients exceeded the minimum threshold of {min_rate:.1E} cm^3/s.")
            continue

        if outliers.empty:
            print(f"  All valid transitions agree within {threshold_pct}%.")
        else:
            for _, row in outliers.iterrows():
                print(f"  Transition {int(row['ilv']):>3} -> {int(row['flv']):<3} | "
                      f"Hybrid: {row[col_hyb]:.3E} | Fitted: {row[col_fit]:.3E} | "
                      f"Diff: {row[f'{T}K_diff_%']:.1f}%")

        # --- Generate Diagonal Plot ---
        plt.figure(figsize=(8, 8))
        plt.scatter(fit_vals[valid_mask], hyb_vals[valid_mask], color='red', alpha=0.6, label='Transitions', zorder=5)

        # Calculate bounds safely now that Infs and NaNs are completely removed
        min_val = min(np.min(fit_vals[valid_mask]), np.min(hyb_vals[valid_mask]))
        max_val = max(np.max(fit_vals[valid_mask]), np.max(hyb_vals[valid_mask]))
        
        lim_min = max(min_val * 0.5, min_rate / 2) 
        lim_max = max_val * 2.0 if max_val > 0 else 1e-10

        line_vals = np.array([lim_min, lim_max])
        plt.plot(line_vals, line_vals, 'k-', label='y = x (Perfect Match)', zorder=4)

        factor = 1.0 + (threshold_pct / 100.0)
        plt.plot(line_vals, line_vals * factor, 'b--', alpha=0.5, label=rf'+/- {threshold_pct}% Bounds', zorder=4)
        plt.plot(line_vals, line_vals / factor, 'b--', alpha=0.5, zorder=4)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.xlabel(r'Fitted Rate Coefficient $k_{fit}$ (cm$^3$/s)', fontsize=14)
        plt.ylabel(r'Hybrid Rate Coefficient $k_{hyb}$ (cm$^3$/s)', fontsize=14)
        
        title_str = f'Hybrid vs Fitted Rates at T = {T} K'
        if max_ilv is not None:
            title_str += f'\n(Initial State $\leq$ {max_ilv}  |  $k \geq$ {min_rate:.1E} | A = {EXTRAP_A_MANUAL_VALUE})'
        else:
            title_str += f'\n($k \geq$ {min_rate:.1E})'
        plt.title(title_str, fontsize=14)
        
        plt.grid(True, which='both', ls='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'Rate_Comparison_{T}K.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return df_merged

def compare_with_scaling_law(calc_txt_path, user_input_path, check_temps, min_rate=1e-15, save_dir=None):
    """
    Reads the calculated rate coefficients, compares them against the scaling law,
    prints a summary table to the console, and plots the results.
    """
    print(f"\nEvaluating Scaling Law Comparison from: {calc_txt_path}")
    
    # 1. Parse states to get the J values for each index
    states_dict = parse_states(user_input_path)
    if not states_dict:
        print("Error: Could not load states for scaling comparison.")
        return

    # 2. Read the calculated rates
    try:
        df_calc = pd.read_csv(calc_txt_path, sep=r'\s+', skiprows=[1])
    except Exception as e:
        print(f"Error reading calculated rates file: {e}")
        return

    # 3. Map the initial state index (ilv) to its actual J' quantum number
    df_calc['J_prime'] = df_calc['ilv'].apply(lambda x: states_dict[x]['J'] if x in states_dict else np.nan)
    df_calc = df_calc.dropna(subset=['J_prime'])

    for T in check_temps:
        col_calc = f"{T}K"
        
        if col_calc not in df_calc.columns:
            print(f"Warning: Temperature {T}K not found in the calculated file.")
            continue
            
        df_calc[col_calc] = pd.to_numeric(df_calc[col_calc], errors='coerce')
        
        k_calc = df_calc[col_calc].values
        J_prime = df_calc['J_prime'].values
        
        # --- THE SCALING EQUATION ---
        k_scale = (1e-11 / (2 * J_prime + 1)) * np.sqrt(T)
        
        # Filter out NaN/Inf and physically negligible rates
        valid_mask = (k_calc >= min_rate) & np.isfinite(k_calc) & np.isfinite(k_scale)
        
        if np.sum(valid_mask) == 0:
            print(f"  No valid rates at {T}K above {min_rate:.1E} cm^3/s to plot.")
            continue

        # Extract the valid data for the table and plot
        df_valid = df_calc[valid_mask]
        valid_calc = k_calc[valid_mask]
        valid_scale = k_scale[valid_mask]

        # --- Print Console Table ---
        print(f"\n" + "="*80)
        print(f" SCALING LAW COMPARISON AT T = {T} K")
        print("="*80)
        header = f"{'State (ilv)':>12} | {'J value':>8} | {'k_MQCT (cm^3/s)':>18} | {'k_Scale (cm^3/s)':>18} | {'Ratio (MQCT/Scale)':>18}"
        print(header)
        print("-" * len(header))
        
        for _, row in df_valid.iterrows():
            state = int(row['ilv'])
            if state == 86:
                j_val = int(row['J_prime'])
                k_m = row[col_calc]
                k_s = (1e-11 / (2 * j_val + 1)) * np.sqrt(T)
                ratio = k_m / k_s
                print(f"{state:>12} | {j_val:>8} | {k_m:>18.3E} | {k_s:>18.3E} | {ratio:>18.2f}")
        print("="*80)

        # --- Generate Diagonal Plot ---
        plt.figure(figsize=(8, 8))
        plt.scatter(valid_scale, valid_calc, color='purple', alpha=0.6, label='Transitions', zorder=5)

        min_val = min(np.min(valid_scale), np.min(valid_calc))
        max_val = max(np.max(valid_scale), np.max(valid_calc))
        
        lim_min = max(min_val * 0.5, min_rate / 2) 
        lim_max = max_val * 2.0 if max_val > 0 else 1e-10

        line_vals = np.array([lim_min, lim_max])
        
        plt.plot(line_vals, line_vals, 'k-', label='y = x (Perfect Match)', zorder=4)
        plt.plot(line_vals, line_vals * 2.0, 'b--', alpha=0.4, label='Factor of 2 Bounds', zorder=4)
        plt.plot(line_vals, line_vals / 2.0, 'b--', alpha=0.4, zorder=4)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        
        plt.xlabel(rf'Scaling Law Rate: $10^{{-11}}\sqrt{{{T}}} /(2J^\prime + 1) / $ (cm$^3$/s)', fontsize=14)
        plt.ylabel(r'Calculated MQCT Rate $k_{calc}$ (cm$^3$/s)', fontsize=14)
        plt.title(rf'MQCT vs Scaling Law at T = {T} K', fontsize=16)
        
        plt.grid(True, which='both', ls='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'Scaling_Comparison_{T}K.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  -> Saved scaling plot: {save_path}")
            plt.close()
        else:
            plt.show()

# =================================================================
# 5. DATABASE PROCESSING & EXPORT
# =================================================================

def calculate_rates_for_database(db_path, user_input_path, temps, cross_sec_method, rate_method):
    print("Reading Database...")
    cols = ["j_1", "ka_1", "kc_1", "j_2", "ka_2", "kc_2", "E_1", "E_2", "U", "sigmaU_qu", "sigmaU_ex", "E_coll_qu", "sigmaE_qu", "E_coll_ex", "sigmaE_ex", "Tot_DeltaE"]
    df_raw = pd.read_csv(db_path, delim_whitespace=True, skiprows=1, names=cols)
    
    # Map ilv and flv directly using user input
    states_dict = parse_states(user_input_path)
    if not states_dict:
        return pd.DataFrame() # Stop if mapping fails
        
    df = map_db_to_user_input(df_raw, states_dict)
    
    results = []
    grouped = df.groupby(['ilv', 'flv', 'j_2', 'j_1', 'E_2', 'E_1'])
    print(f"Calculating rates (CS: '{cross_sec_method}', Rate: '{rate_method}') for {len(grouped)} transitions...")
    
    for (ilv, flv, j_2, j_1, e_2, e_1), group_data in tqdm(grouped, desc="Calculating Rate Co effs"):
        group_data = group_data.sort_values('U')
        U_raw = group_data['U'].values
        
        #print(f"Calculation for Inital State : {ilv}")
        
        # --- SELECT CROSS SECTION METHOD ---
        sig_dir = group_data['sigmaU_qu'].values
        sig_der = group_data['sigmaU_ex'].values * ((2 * j_1 + 1) / (2 * j_2 + 1))
        
        if cross_sec_method == "direct":
            sigma_raw = sig_dir
        elif cross_sec_method == "derived":
            sigma_raw = sig_der
        elif cross_sec_method == "average":
            sigma_raw = (sig_dir + sig_der) / 2.0
        else:
            raise ValueError("Invalid CROSS_SECTION_METHOD. Use 'direct', 'derived', or 'average'.")

        dE_real = e_1 - e_2  # Quenching
        U0 = abs(dE_real) / 4.0
        g_i = 2 * j_2 + 1  # Upper state degeneracy
        
        q_fn, q_low, q_high = get_sigma_func_with_bounds(U_raw, sigma_raw, U0)
        
        mask_low = U_raw < EXTRAP_MIN_U
        u_raw_low, s_raw_low = U_raw[mask_low], sigma_raw[mask_low]
        
        row_data = {'ilv': ilv, 'flv': flv}
        
        q_rates_fitted, q_rates_hybrid = [], []
        integrand_funcs_map = {}

        for T in temps:
            kT = K_TO_WN * T
            pref = ((RATE_CONSTANT * np.sqrt(T)) / (kT**2)) * (np.exp(abs(dE_real)/(2*kT))/g_i)
            
            def make_integrand_func(temp_k, u_val):
                kt_val = K_TO_WN * temp_k
                term = (abs(dE_real)/(4*max(u_val, 1e-9)))**2
                return g_i * u_val * q_fn(u_val) * (1.0 - term) * np.exp(-u_val/kt_val * (1 + term))
            
            # Hybrid Calculation
            integral_part1 = 0.0
            if len(u_raw_low) > 1:
                integrand_vals = []
                for u_val, s_val in zip(u_raw_low, s_raw_low):
                    u_safe = max(u_val, 1e-9)
                    term = (abs(dE_real)/(4*u_safe))**2
                    val = g_i * u_safe * s_val * (1.0 - term) * np.exp(-u_safe/kT * (1 + term))
                    integrand_vals.append(val)
                integral_part1 = np.trapz(integrand_vals, u_raw_low)
            
            q_r_part2, _ = quad(lambda u: make_integrand_func(T, u), EXTRAP_MIN_U, 15000.0, limit=200, epsabs=1e-35)
            q_r_fit, _   = quad(lambda u: make_integrand_func(T, u), U0, 15000.0, limit=200, epsabs=1e-35)
            
            rate_hybrid = pref * (integral_part1 + q_r_part2)
            rate_fitted = pref * q_r_fit
            
            # --- SELECT RATE CALCULATION METHOD ---
            if rate_method == "hybrid":
                row_data[f"{T}K"] = rate_hybrid
            elif rate_method == "fitted":
                row_data[f"{T}K"] = rate_fitted
            else:
                raise ValueError("Invalid RATE_METHOD. Use 'hybrid' or 'fitted'.")
                
            q_rates_hybrid.append(rate_hybrid)
            q_rates_fitted.append(rate_fitted)
            
            if T in VISUALIZE_TEMPS:
                integrand_funcs_map[T] = lambda u, t=T: make_integrand_func(t, u)
                
        results.append(row_data)
        
        # Trigger plot if this is the targeted transition
        if (ilv, flv) in VIZ_TRANSITION:
            plot_detailed_transition(ilv, flv, dE_real, g_i, U_raw, sigma_raw, q_fn, q_low, q_high, U0, 
                                     temps, VISUALIZE_TEMPS, q_rates_fitted, q_rates_hybrid, integrand_funcs_map)

    df_rates = pd.DataFrame(results)
    
    # NEW: Sorting logically from 2->1, 3->1, 3->2 upwards
    df_rates = df_rates.sort_values(by=['ilv', 'flv'], ascending=[True, True]).reset_index(drop=True)
    print(f"Calculation done for State: {ilv}")
    return df_rates

def export_to_txt(df_rates, output_path, temps):
    """Exports the DataFrame to a nicely formatted text file matching the written notebook."""
    with open(output_path, 'w') as f:
        # Create Header
        header = f"{'ilv':>5} {'flv':>5} " + " ".join([f"{T}K".rjust(13) for T in temps]) + "\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        
        # Write rows
        for _, row in df_rates.iterrows():
            line = f"{int(row['ilv']):>5} {int(row['flv']):>5} "
            line += " ".join([f"{row[f'{T}K']:>13.4E}" for T in temps]) + "\n"
            f.write(line)
            
    print(f"\nRate coefficients successfully exported as text file to: {output_path}")






# # ===============================================================
# # 6.1. Calcualate Rate Co-efficients
# # ===============================================================


# if __name__ == "__main__":

#     # Which transition do you want to plot the detailed 2x3 grid for? (ilv, flv)
#     # VIZ_TRANSITION = [(17, 12),(18,16),(14,11),(18,11),(10,4)] # bad ones (small)
#     # VIZ_TRANSITION = [(18, 13),(7,4),(17,14),(16,8),(8,2)] # good ones ones (Large)
#     VIZ_TRANSITION = [(17, 12),(18,16),(14,11),(18,11),(10,4),(18, 13),(7,4),(17,14),(16,8),(8,2)]
#     if os.path.exists(DB_FILE_PATH) and os.path.exists(USER_INPUT_PATH):
#         rate_matrix = calculate_rates_for_database(DB_FILE_PATH, USER_INPUT_PATH, TEMPS, CROSS_SECTION_METHOD, RATE_METHOD)
#         if not rate_matrix.empty:
#             export_to_txt(rate_matrix, OUTPUT_TXT_PATH, TEMPS)
#     else:
#         print(f"Error: Check paths. DB exists: {os.path.exists(DB_FILE_PATH)}, User Input exists: {os.path.exists(USER_INPUT_PATH)}")

# # =================================================================
# # 6.2. Analyse Data - Prints Difference in Hybrid and fitted methods
# # =================================================================
# if __name__ == "__main__":
#     # Point these to the two separate text files you generated
#     HYBRID_FILE = r"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/Rate_Coefficients_full_range_hybrid_U_min_40.dat"
#     FITTED_FILE = rf"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/Rate_Coefficients_full_range_fit_U_min_40.dat"
    
#     TEMPS_TO_CHECK = [10,50,100]
#     THRESHOLD = 200  # Print anything with > difference
    
#     MAX_INITIAL_STATE = 86 
    
#     # Filter out any rates below this value
#     MINIMUM_RATE = 1e-12
    
#     PLOT_DIR = None   
    
#     df_analysis = analyze_saved_rate_differences(
#         hybrid_txt_path=HYBRID_FILE, 
#         fitted_txt_path=FITTED_FILE, 
#         check_temps=TEMPS_TO_CHECK, 
#         threshold_pct=THRESHOLD, 
#         max_ilv=MAX_INITIAL_STATE,
#         min_rate=MINIMUM_RATE,
#         save_dir=PLOT_DIR
#     )

# =================================================================
# 6.3. Analyse Data - Scaling Law Comparison
# =================================================================
if __name__ == "__main__":
    
    # Use the file your script just exported!
    CALCULATED_FILE = rf"C:/Users/4374vijayv/OneDrive - Marquette University/Ethanimine/Ethanimine_E_He/2026/Apr/Cross_Sections_CC/Rate_Coefficients_full_range_fit_U_min_40.dat"
    
    TEMPS_TO_CHECK = [10, 50, 100, 300]
    MINIMUM_RATE = 1e-15
    PLOT_DIR = None  # Change to a folder path if you want them saved
    
    # Run the function
    compare_with_scaling_law(
        calc_txt_path=CALCULATED_FILE,
        user_input_path=USER_INPUT_PATH,
        check_temps=TEMPS_TO_CHECK,
        min_rate=MINIMUM_RATE,
        save_dir=PLOT_DIR
    )