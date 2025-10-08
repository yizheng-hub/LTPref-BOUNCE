#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script analyzes the simulation results from the main dialogue framework,
runs a Home Energy Management System (HEMS) simulation for each successful user,
and generates performance comparison plots in a publication-ready format.

The script performs the following steps:
1.  Loads pre-processed 24-hour environmental data (temperature, price, etc.).
2.  Loads and filters user dialogue simulation results.
3.  For each user, runs 'Personalized' and 'Baseline' HEMS simulations.
4.  Saves the aggregated HEMS simulation results to a CSV file.
5.  Performs a granular energy analysis by user preference group.
6.  Generates and saves two summary plots:
    a) A 2x2 boxplot comparing performance metrics (Fig 4).
    b) A 1x3 line chart showing temperature trajectories for representative users (Fig S).
"""

import json
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import seaborn as sns
from pythermalcomfort.models import pmv_ppd_ashrae
from tqdm import tqdm
from scipy.stats import mannwhitneyu

# ============================================================================
# --- 1. GLOBAL CONFIGURATION ---
# ============================================================================

# --- Input File Paths ---
# Assumes this script is run from the project's 'src' directory
RESULTS_FOLDER = os.path.join("..", "output", "full_simulation_results_output")
ENVIRONMENT_DATA_FILE = os.path.join("..", "data", "environment_data_HK_20240715.csv")

# --- Output File Paths ---
# Folders 'results' and 'plots' will be created at the project root
RESULTS_DIR = os.path.join("..", "results")
PLOTS_DIR = os.path.join("..", "plots")
HEMS_RESULTS_OUTPUT_FILE = os.path.join(RESULTS_DIR, "hems_simulation_results.csv")
FINAL_FIGURE_OUTPUT_FILE = os.path.join(PLOTS_DIR, "Fig.7.png")
TRAJECTORY_FIGURE_OUTPUT_FILE = os.path.join(PLOTS_DIR, "Fig.S3.png")

# --- HEMS Simulation Parameters ---
BUILDING_PARAMS = {'R_eq': 2.0, 'C_eq': 10.0}
HVAC_PARAMS = {'eta': 2.5, 'P_max': 3.0}
PENALTY_COEFFICIENT = 1000.0
BASELINE_TEMP_RANGE = (24.0, 26.0)
SUMMER_CLO = 0.5
SUMMER_MET = 1.0


# ============================================================================
# --- 2. HELPER AND CORE HEMS FUNCTIONS ---
# ============================================================================

def calculate_pmv(air_temp, mrt, rh, air_vel, clo, met):
    """Calculates PMV using the pythermalcomfort library."""
    try:
        return pmv_ppd_ashrae(tdb=air_temp, tr=mrt, vr=air_vel, rh=rh, clo=clo, met=met).pmv
    except Exception:
        return None

def find_temp_for_pmv(target_pmv, clo, met, rh=50, air_vel=0.1):
    """Finds the operative temperature that results in a target PMV."""
    def temp_diff_function(temp):
        calculated_pmv = calculate_pmv(temp, temp, rh, air_vel, clo, met)
        return calculated_pmv - target_pmv if calculated_pmv is not None else 999
    try:
        return brentq(temp_diff_function, 10, 40)
    except Exception:
        return None

def convert_pmv_range_to_temp_range(pmv_min, pmv_max, clo, met):
    """Converts a PMV range to a corresponding temperature range."""
    t_min = find_temp_for_pmv(pmv_min, clo, met)
    t_max = find_temp_for_pmv(pmv_max, clo, met)
    if t_min is None or t_max is None:
        return (None, None)
    return (min(t_min, t_max), max(t_min, t_max))

def run_hems_simulation(
    comfort_temp_range: tuple[float, float], outdoor_temps: np.ndarray,
    electricity_prices: np.ndarray, carbon_intensity: np.ndarray,
    solar_gains: np.ndarray, internal_gains: np.ndarray,
    building_params: dict, hvac_params: dict, penalty_coeff: float
) -> dict:
    """Runs a 24-hour HEMS optimization."""
    # (This function remains unchanged from your previous version)
    T_low, T_high = comfort_temp_range
    T = len(outdoor_temps)
    delta_t = 1.0
    R_eq, C_eq = building_params['R_eq'], building_params['C_eq']
    eta, P_max = hvac_params['eta'], hvac_params['P_max']
    def objective_function(u_control: np.ndarray) -> float:
        T_in = np.zeros(T + 1); T_in[0] = outdoor_temps[0]
        P_electric = u_control * P_max; P_thermal = -eta * P_electric
        total_cost, total_violation = 0.0, 0.0
        for t in range(T):
            exp_term = np.exp(-delta_t / (R_eq * C_eq))
            total_heat_input = P_thermal[t] + solar_gains[t] + internal_gains[t]
            T_in[t+1] = T_in[t] * exp_term + (outdoor_temps[t] + R_eq * total_heat_input) * (1 - exp_term)
            total_cost += electricity_prices[t] * P_electric[t] * delta_t
            total_violation += max(0, T_in[t+1] - T_high) + max(0, T_low - T_in[t+1])
        return total_cost + penalty_coeff * total_violation
    result = minimize(fun=objective_function, x0=np.full(T, 0.5), method='SLSQP', bounds=[(0, 1) for _ in range(T)], options={'maxiter': 200})
    if not result.success: tqdm.write(f"Warning: Opt did not converge for range {comfort_temp_range}.")
    u_optimal = result.x; P_electric_optimal = u_optimal * P_max; P_thermal_optimal = -eta * P_electric_optimal
    T_in_optimal = np.zeros(T + 1); T_in_optimal[0] = outdoor_temps[0]
    total_energy_cost, total_cvi, total_carbon_emissions = 0.0, 0.0, 0.0
    for t in range(T):
        exp_term = np.exp(-delta_t / (R_eq * C_eq)); total_heat_input_optimal = P_thermal_optimal[t] + solar_gains[t] + internal_gains[t]
        T_in_optimal[t+1] = T_in_optimal[t] * exp_term + (outdoor_temps[t] + R_eq * total_heat_input_optimal) * (1 - exp_term)
        energy_consumed_t = P_electric_optimal[t] * delta_t
        total_energy_cost += electricity_prices[t] * energy_consumed_t
        total_carbon_emissions += carbon_intensity[t] * energy_consumed_t
        total_cvi += (max(0, T_in_optimal[t+1] - T_high) + max(0, T_low - T_in_optimal[t+1])) * delta_t
    total_energy_consumed = np.sum(P_electric_optimal * delta_t)
    return { 'total_energy_cost': total_energy_cost, 'total_cvi': total_cvi, 'total_carbon_emissions': total_carbon_emissions / 1000, 'total_energy_consumed_kWh': total_energy_consumed, 'T_in_optimal': T_in_optimal.tolist() }

def add_p_value_annotation(ax, df, col1, col2, **kwargs):
    """Adds Mann-Whitney U test p-value annotation to a boxplot."""
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()
    if len(data1) < 1 or len(data2) < 1: return
    try:
        _, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    except ValueError:
        p_value = 1.0 # Handle case with insufficient data
    
    p_text = "P < 0.001" if p_value < 0.001 else f"P = {p_value:.2f}"
    
    y_max = max(data1.max(), data2.max())
    ax.plot([0, 0, 1, 1], [y_max*1.05, y_max*1.1, y_max*1.1, y_max*1.05], lw=1.5, c='black')
    ax.text(0.5, y_max*1.12, p_text, ha='center', va='bottom', color='black', **kwargs)


# ============================================================================
# --- 3. MAIN EXECUTION LOGIC ---
# ============================================================================

def main():
    """Main function to run the entire analysis pipeline."""
    print("--- HEMS Analysis and Plotting Script Started ---")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Ensured output directories '{os.path.basename(RESULTS_DIR)}' and '{os.path.basename(PLOTS_DIR)}' exist.")

    # --- Load Environment Data ---
    print(f"Loading environment data from '{ENVIRONMENT_DATA_FILE}'...")
    try:
        env_df = pd.read_csv(ENVIRONMENT_DATA_FILE)
        outdoor_temps = env_df['Outdoor_Temp_C'].values
        electricity_prices = env_df['Price_HKD_per_kWh'].values
        carbon_intensity = env_df['Carbon_Intensity_gCO2_per_kWh'].values
        solar_gains = env_df['Q_solar_kW'].values
        internal_gains = env_df['Q_internal_kW'].values
    except FileNotFoundError:
        print(f"FATAL: Environment data file not found at '{ENVIRONMENT_DATA_FILE}'. Exiting."); exit()

    # --- Load and Process Simulation Results ---
    print(f"\nLoading and filtering user simulation results from '{RESULTS_FOLDER}'...")
    all_user_results = []
    # (File loading logic remains the same)
    try: all_files_in_folder = os.listdir(RESULTS_FOLDER)
    except FileNotFoundError: print(f"FATAL: Results folder '{RESULTS_FOLDER}' not found."); exit()
    for filename in tqdm(all_files_in_folder, desc="1. Loading JSON files"):
        if filename.endswith(".json") and "_precise_" in filename and filename.startswith("results_user_"):
            file_path = os.path.join(RESULTS_FOLDER, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                if (data.get('user_behavior_type') == 'precise' and data.get('pmv_inference_successful') is True and data.get('inferred_min_pmv') is not None):
                    all_user_results.append(data)
            except Exception as e: tqdm.write(f"Warning: Error processing file {filename}: {e}")
    if not all_user_results: print("FATAL: No successful 'precise' user results found for HEMS simulation."); exit()
    print(f"Found {len(all_user_results)} successful 'precise' user profiles.")

    # --- Run HEMS Simulations ---
    hems_results_list = []
    print("\nRunning HEMS simulations for all users...")
    # (HEMS simulation loop remains the same)
    for user_data in tqdm(all_user_results, desc="2. HEMS Simulations"):
        user_id = user_data['user_id']
        inferred_pmv_range = (user_data['inferred_min_pmv'], user_data['inferred_max_pmv'])
        personalized_temp_range = convert_pmv_range_to_temp_range(inferred_pmv_range[0], inferred_pmv_range[1], SUMMER_CLO, SUMMER_MET)
        if personalized_temp_range[0] is None: continue
        hems_personalized = run_hems_simulation(personalized_temp_range, outdoor_temps, electricity_prices, carbon_intensity, solar_gains, internal_gains, BUILDING_PARAMS, HVAC_PARAMS, PENALTY_COEFFICIENT)
        hems_baseline = run_hems_simulation(BASELINE_TEMP_RANGE, outdoor_temps, electricity_prices, carbon_intensity, solar_gains, internal_gains, BUILDING_PARAMS, HVAC_PARAMS, PENALTY_COEFFICIENT)
        T_low_user, T_high_user = personalized_temp_range
        T_in_personalized = np.array(hems_personalized['T_in_optimal']); T_in_baseline = np.array(hems_baseline['T_in_optimal'])
        cvi_personalized_vs_user_target = np.sum(np.maximum(0, T_in_personalized[1:] - T_high_user) + np.maximum(0, T_low_user - T_in_personalized[1:]))
        cvi_baseline_vs_user_target = np.sum(np.maximum(0, T_in_baseline[1:] - T_high_user) + np.maximum(0, T_low_user - T_in_baseline[1:]))
        hems_results_list.append({
            'user_id': user_id, 'true_ideal_pmv': user_data['true_ideal_pmv'],
            'personalized_t_low': personalized_temp_range[0], 'personalized_t_high': personalized_temp_range[1],
            'cost_personalized': hems_personalized['total_energy_cost'], 'cvi_personalized': cvi_personalized_vs_user_target,
            'carbon_personalized_kg': hems_personalized['total_carbon_emissions'], 'energy_personalized_kWh': hems_personalized['total_energy_consumed_kWh'],
            'cost_baseline': hems_baseline['total_energy_cost'], 'cvi_baseline': cvi_baseline_vs_user_target,
            'carbon_baseline_kg': hems_baseline['total_carbon_emissions'], 'energy_baseline_kWh': hems_baseline['total_energy_consumed_kWh'],
            'T_in_personalized': T_in_personalized.tolist(), 'T_in_baseline': T_in_baseline.tolist()
        })

    hems_df = pd.DataFrame(hems_results_list)
    hems_df.to_csv(HEMS_RESULTS_OUTPUT_FILE, index=False)
    print(f"\nHEMS simulation results for {len(hems_df)} users saved to '{HEMS_RESULTS_OUTPUT_FILE}'.")

    # --- Granular Energy Analysis by Preference Group ---
    print("\n--- Performing Granular Energy Analysis by Preference Group ---")
    def assign_preference_group(pmv):
        if pmv <= -0.5: return 'Cool-seeking'
        elif pmv >= 0.5: return 'Warm-seeking'
        else: return 'Neutral'
    hems_df['preference_group'] = hems_df['true_ideal_pmv'].apply(assign_preference_group)
    grouped_energy_analysis = hems_df.groupby('preference_group').agg(
        user_count=('user_id', 'count'),
        Avg_E_baseline_kWh=('energy_baseline_kWh', 'mean'),
        Avg_E_personalized_kWh=('energy_personalized_kWh', 'mean')
    ).round(4)
    print("\n" + "="*70); print("--- Granular Energy Consumption Analysis (in kWh) ---"); print("="*70)
    print(grouped_energy_analysis); print("="*70)
    print("\n--- Key Numbers for Your Paper ---")
    if 'Warm-seeking' in grouped_energy_analysis.index and 'Neutral' in grouped_energy_analysis.index:
        avg_e_baseline_warm = grouped_energy_analysis.loc['Warm-seeking', 'Avg_E_baseline_kWh']
        avg_e_baseline_neutral = grouped_energy_analysis.loc['Neutral', 'Avg_E_baseline_kWh']
        percentage_diff = ((avg_e_baseline_warm - avg_e_baseline_neutral) / avg_e_baseline_neutral) * 100
        print(f"Warm-seeking users consumed {percentage_diff:.1f}% MORE energy than Neutral users under the baseline.")
    print("="*70)

    # --- Plotting Section ---
    print("\n--- Generating Final Figures ---")
    try:
        plt.rcParams['font.family'] = 'serif'; plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 12
        sns.set_theme(style="white", rc=plt.rcParams)
        print("Font set to Times New Roman for all plots.")
    except Exception as e:
        print(f"Warning: Could not set Times New Roman font. Using default. Error: {e}")
        sns.set_theme(style="white")

    # --- Generate 2x2 Figure 7 ---
    print("\nGenerating 2x2 Figure 7...")
    palette = {"Baseline": sns.color_palette("Greens_d", 3)[1], "LTPref-BOUNCE": sns.color_palette("Blues_d", 3)[2]}
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
    metrics_config = {
        'Energy Cost': {'ax': axes4[0, 0], 'base_col': 'cost_baseline', 'pers_col': 'cost_personalized', 'ylabel': 'Average Daily Value ($ HKD)', 'panel_label': '(a)'},
        'Carbon Emissions': {'ax': axes4[0, 1], 'base_col': 'carbon_baseline_kg', 'pers_col': 'carbon_personalized_kg', 'ylabel': 'Average Daily Value (kg CO$_2$)', 'panel_label': '(b)'},
        'Energy Consumption': {'ax': axes4[1, 0], 'base_col': 'energy_baseline_kWh', 'pers_col': 'energy_personalized_kWh', 'ylabel': 'Average Daily Value (kWh)', 'panel_label': '(c)'},
        'Comfort Violation': {'ax': axes4[1, 1], 'base_col': 'cvi_baseline', 'pers_col': 'cvi_personalized', 'ylabel': 'Average Daily Value (°C·h)', 'panel_label': '(d)'}
    }
    for metric, config in metrics_config.items():
        ax = config['ax']; base_col, pers_col = config['base_col'], config['pers_col']
        plot_df = pd.melt(hems_df[[base_col, pers_col]], var_name='Group', value_name='Value')
        plot_df['Group'] = plot_df['Group'].map({base_col: 'Baseline', pers_col: 'LTPref-BOUNCE'})
        sns.boxplot(data=plot_df, x='Group', y='Value', ax=ax, palette=palette, width=0.5, showfliers=False, boxprops=dict(alpha=0.8))
        sns.stripplot(data=plot_df, x='Group', y='Value', ax=ax, jitter=True, palette=palette, alpha=0.5, s=4)
        add_p_value_annotation(ax, hems_df, base_col, pers_col, fontsize=12)
        ax.set_title(metric, fontsize=14, weight='bold'); ax.set_xlabel(''); ax.set_ylabel(config['ylabel'])
        ax.tick_params(axis='x', labelsize=12); ax.set_ylim(bottom=0)
        ax.text(-0.1, 1.05, config['panel_label'], transform=ax.transAxes, size=16, weight='bold')
    sns.despine(fig=fig4)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(FINAL_FIGURE_OUTPUT_FILE, dpi=300)
    print(f"Figure 7 saved to '{FINAL_FIGURE_OUTPUT_FILE}'")
    plt.show()

    # --- Generate 1x3 Supplementary Figure ---
    print("\nGenerating Supplementary Figure for HEMS Trajectories...")
    try:
        representative_users = {
            'Cool-seeking User': hems_df[hems_df['preference_group'] == 'Cool-seeking'].iloc[0],
            'Neutral User': hems_df[hems_df['preference_group'] == 'Neutral'].iloc[0],
            'Warm-seeking User': hems_df[hems_df['preference_group'] == 'Warm-seeking'].iloc[0]
        }
    except IndexError:
        print("Warning: Could not find users for all three preference groups. Skipping trajectory plot."); return
    figS, axesS = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, (group_name, user_data) in enumerate(representative_users.items()):
        ax = axesS[i]
        T_in_personalized = np.array(user_data['T_in_personalized']); T_in_baseline = np.array(user_data['T_in_baseline'])
        personalized_zone = (user_data['personalized_t_low'], user_data['personalized_t_high'])
        hours = np.arange(25)
        ax.plot(hours, T_in_personalized, label='LTPref-BOUNCE (Personalized)', color=palette["LTPref-BOUNCE"], lw=2)
        ax.plot(hours, T_in_baseline, label='Baseline (Non-personalized)', color=palette["Baseline"], lw=2)
        ax.plot(np.arange(24), outdoor_temps, label='Outdoor Temperature', color='gray', linestyle='--', lw=1.5)
        ax.fill_between(hours, BASELINE_TEMP_RANGE[0], BASELINE_TEMP_RANGE[1], color=palette["Baseline"], alpha=0.15, label=f'Baseline Zone ({BASELINE_TEMP_RANGE[0]:.1f}-{BASELINE_TEMP_RANGE[1]:.1f}°C)')
        ax.fill_between(hours, personalized_zone[0], personalized_zone[1], color=palette["LTPref-BOUNCE"], alpha=0.2, label=f'Personalized Zone ({personalized_zone[0]:.1f}-{personalized_zone[1]:.1f}°C)')
        ax.set_title(f"{group_name}\n(ID: {user_data['user_id']})", fontsize=12, weight='bold')
        ax.set_xlabel('Time of Day (Hour)', fontsize=12)
        if i == 0: ax.set_ylabel('Temperature (°C)', fontsize=12)
    sns.despine(fig=figS)
    handles, labels = axesS[0].get_legend_handles_labels()
    figS.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False, fontsize=12)
    plt.tight_layout()
    figS.subplots_adjust(bottom=0.2)
    plt.savefig(TRAJECTORY_FIGURE_OUTPUT_FILE, dpi=600, bbox_inches='tight')
    print(f"Supplementary trajectory figure saved to '{TRAJECTORY_FIGURE_OUTPUT_FILE}'")
    plt.show()

if __name__ == "__main__":
    main()