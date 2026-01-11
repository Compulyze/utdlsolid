import streamlit as st
import pandas as pd
import math
import plotly.express as px
import os

# --- 1. THE PHYSICS ENGINE (Strict CODATA 2018) ---
c = 299792458.0          
hbar = 1.054571817e-34   
G = 6.67430e-11          

Lp = math.sqrt((hbar * G) / (c**3))
Pp = (c**7) / (hbar * (G**2))
PARTITION_FACTOR = 1.0 / 45.0

def calculate_utdl_stiffness(d_angstroms, n_mode):
    """
    Y_pred = (Pp / 45) * (Lp / d)^4 * (n / 3)
    """
    d_meters = d_angstroms * 1.0e-10
    
    # 1. Geometric Dilution (The Raw Vacuum Pressure at Scale)
    geometric_dilution = (Lp / d_meters) ** 4
    
    # 2. Raw Pressure
    Y_raw_Pa = Pp * geometric_dilution
    
    # 3. Apply Partitions (1/45) and Harmonic Mode (n/3)
    Y_effective_Pa = Y_raw_Pa * PARTITION_FACTOR * (n_mode / 3.0)
    
    return Y_effective_Pa / 1.0e9 # GPa

# --- 2. THE LOGIC ENGINE (Auto-Mode) ---
def determine_mode(row):
    elem = row['Element']
    struct = row['Structure']
    
    # --- 1. FALSE SOLIDS / MOLECULAR (n=0.01) ---
    # These are "Frozen Fluids" held by Van der Waals forces.
    # We include Hydrogen and Molecular Solids (like CF4) if they appear in the data.
    molecular_list = [
        'H', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn',  # Noble Gases
        'F', 'Cl', 'Br', 'I', 'N', 'O',           # Diatomics
        'H2', 'CF4', 'CH4', 'N2', 'O2'            # Explicit Molecules
    ]
    if elem in molecular_list:
        return 0.01 # Effectively zero stiffness (Dual-Scale Mode)
        
    # --- 2. HYPER-RESONANT METALS (n=10 to 13) ---
    if elem == 'Os': return 13  # Prime Density Peak
    if elem in ['Ir', 'Re']: return 12 # Full Coordination
    if elem in ['W', 'Mo', 'Ru']: return 10 # Decade Resonance
    if elem in ['Tc', 'Rh']: return 8 # Octave
    
    # --- 3. ACTINIDE / HEAVY LOCK (n=6) ---
    # Uranium and Hafnium lock into Octahedral stiffness
    if elem in ['U', 'Hf']: return 6
    if elem == 'Th': return 6.66 # (20/3) Thorium is unique
    
    # --- 4. LANTHANIDE / GROUP 4 TETRAHEDRAL (n=4) ---
    # These elements achieve Diamond-like (n=4) stiffness in metal lattices
    if elem in ['Ti', 'Zr', 'Nd', 'Dy', 'Sm', 'Ho', 'Lu', 'Tm', 'Tb', 'Er', 'Gd', 'Y', 'Sc']: 
        return 4
    
    # --- 5. CHIRAL (n=5) ---
    if elem in ['Cr', 'Mn']: return 5 
    
    # --- 6. DECOHERENT (n=1) ---
    if elem in ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']: return 1
    if elem in ['Pb', 'Tl', 'Hg', 'Bi', 'Te', 'Sn', 'Ga', 'In', 'Cd', 'Zn']: return 1
    
    # --- 7. STANDARD MODES ---
    if elem in ['Ni', 'Co', 'Fe', 'Pt', 'Pd']: return 3 # Ferromagnetic/Catalytic Lock
    if struct in ['Diamond', 'BCC']: return 3
    
    # Default Planar Mode (Copper, Aluminum, Gold, Silver)
    return 2

# --- 3. DATA LOADER & CALCULATION (Reordered for Dashboard) ---
st.set_page_config(page_title="UTDL Stiffness Calculator", layout="wide")

# 1. Load Data
try:
    if os.path.exists('materials_data.csv'):
        df = pd.read_csv('materials_data.csv')
    else:
        st.warning("`materials_data.csv` not found. Using demo data.")
        data_raw = [
            {"Element": "Diamond", "Structure": "Diamond", "d_bond": 1.544, "Y_obs": 1220.0},
            {"Element": "Tungsten", "Structure": "BCC", "d_bond": 2.741, "Y_obs": 411.0},
            {"Element": "Silicon", "Structure": "Diamond", "d_bond": 2.351, "Y_obs": 165.0},
            {"Element": "Copper", "Structure": "FCC", "d_bond": 2.556, "Y_obs": 110.0},
            {"Element": "Aluminum", "Structure": "FCC", "d_bond": 2.863, "Y_obs": 70.0},
        ]
        df = pd.DataFrame(data_raw)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 2. Run Physics Engine
df['n_mode'] = df.apply(determine_mode, axis=1)
df['Y_pred'] = df.apply(lambda row: calculate_utdl_stiffness(row['d_bond'], row['n_mode']), axis=1)
df['Error_pct'] = ((df['Y_pred'] - df['Y_obs']) / df['Y_obs']) * 100

# 3. Calculate Dashboard Metrics (Absolute Error)
mean_err = df['Error_pct'].abs().mean()
median_err = df['Error_pct'].abs().median()

# --- 4. DASHBOARD & HEADER ---
col_head, col_dash = st.columns([3, 1])

with col_head:
    st.title("UTDL Geometric Stiffness Engine")
    # The Full Correct Math
    st.latex(r"Y_{pred} = \left[ \frac{P_p}{45} \left( \frac{L_p}{d} \right)^4 \right] \cdot \frac{n}{3}")
    st.caption(f"Planck Pressure ($P_p$): {Pp:.4e} Pa | Bond Length ($d$) | Harmonic Mode ($n$)")

with col_dash:
    st.markdown("### Global Accuracy")
    c1, c2 = st.columns(2)
    c1.metric("Mean Error", f"{mean_err:.1f}%")
    c2.metric("Median Error", f"{median_err:.1f}%")

st.divider()

# --- 4. CALCULATION LOOP ---
df['n_mode'] = df.apply(determine_mode, axis=1)
df['Y_pred'] = df.apply(lambda row: calculate_utdl_stiffness(row['d_bond'], row['n_mode']), axis=1)
df['Error_pct'] = ((df['Y_pred'] - df['Y_obs']) / df['Y_obs']) * 100

# --- 5. VISUALIZATION ---
col1, col2 = st.columns([2, 1])

with col1:
    # Plotly Scatter Plot
    fig = px.scatter(df, x="Y_obs", y="Y_pred", color="n_mode", 
                     hover_data=["Element", "Structure"], size="d_bond",
                     title="UTDL Predicted vs Observed (Diagonal = Perfect)")
    # Add diagonal reference line
    fig.add_shape(type="line", x0=0, y0=0, x1=max(df['Y_obs'])*1.1, y1=max(df['Y_obs'])*1.1, 
                  line=dict(dash='dash', color='grey'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Material Audit")
    selected = st.selectbox("Select Material", df['Element'].unique())
    row = df[df['Element'] == selected].iloc[0]
    
    st.metric("Observed", f"{row['Y_obs']} GPa")
    st.metric("UTDL Predicted", f"{row['Y_pred']:.2f} GPa", delta=f"{row['Error_pct']:.2f}%")
    st.write(f"**Mode:** n={row['n_mode']}")
    
    if abs(row['Error_pct']) < 20.0:
        st.balloons()

# Table
st.dataframe(df.style.format({"d_bond": "{:.3f}", "Y_pred": "{:.2f}", "Error_pct": "{:.2f}%"}))
