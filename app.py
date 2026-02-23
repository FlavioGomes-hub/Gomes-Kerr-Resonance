import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from astropy import constants as const
from astropy import units as u
import plotly.graph_objects as go
import time
import os
import networkx as nx 

# --- GOMES CONSTANTS (PHYSICS CALCULATION) ---
# 1. Age of the Universe (Hubble Time)
# Planck 2018 value: ~13.798 Billion years
AGE_UNIVERSE = 13.798e9 * u.yr

# 2. Salpeter Time (t_sal)
# Characteristic timescale for accretion (Eddington limit)
# Formula: (sigma_T * c * epsilon) / (4 * pi * G * m_p)
# Epsilon (Radiative efficiency) = 1.0 (Standard for Black Holes)
epsilon = 1.0
t_salpeter = (epsilon * const.sigma_T * const.c) / (4 * np.pi * const.G * const.m_p)

# 3. EXACT Coupling Factor (Kappa)
# Should yield approximately 30.6 (float64 precision)
GOMES_COUPLING = (AGE_UNIVERSE / t_salpeter).decompose().value

# Console verification
print(f"🔬 CALCULATED KAPPA CONSTANT: {GOMES_COUPLING:.5f}")

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Gomes Cosmological Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark CSS
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #21262D;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #58A6FF;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2EA043;
    }
</style>
""", unsafe_allow_html=True)


# --- CLASS DEFINITIONS ---

class PulsarData:
    """Handles data ingestion and cleaning."""
    
    def __init__(self, source):
        self.raw_data = None
        self.time = None
        self.residuals = None
        self.error = None
        self.filename = "Unknown"
        self.valid = False
        
        if source:
            if isinstance(source, str):
                self.filename = os.path.basename(source)
            else:
                self.filename = source.name
                
            self._load_data(source)

    def _load_data(self, source):
        try:
            # Robust parsing for NANOGrav / Tempo2 style files
            # Forced float64 precision
            
            # Note: pd.read_csv handles both file paths and buffers
            df = pd.read_csv(
                source, 
                sep=r'\s+', 
                comment='#', 
                header=None, 
                on_bad_lines='skip',
                dtype={0: 'float64', 1: 'float64', 2: 'float64'}
            )
            
            # Check for at least 3 columns
            if df.shape[1] < 3:
                # Fallback logic for CSV
                if hasattr(source, 'seek'):
                     source.seek(0)
                try:
                    df_fallback = pd.read_csv(source) 
                    # Note: We let pandas infer float64 by default for numeric CSVs
                    if df_fallback.shape[1] >= 2:
                        df = df_fallback
                        # Map likely columns for fallback dummy data
                        if 'time_sec' in df.columns:
                            self.time = df['time_sec'].astype('float64').values
                        else:
                            self.time = df.iloc[:, 0].astype('float64').values
                            
                        if 'residuals_sec' in df.columns:
                            self.residuals = df['residuals_sec'].astype('float64').values
                        else:
                            self.residuals = df.iloc[:, 1].astype('float64').values
                            
                        self.error = np.ones_like(self.residuals) * 1e-6 # Dummy error
                        self.valid = True
                        self._clean()
                        return
                except:
                    pass
                
                st.error(f"Invalid format: The file must contain at least 3 columns (MJD, Residuals, Error).")
                return

            # Standards Extraction (NANOGrav usually)
            self.time = df.iloc[:, 0].values * 86400.0  # MJD to Seconds (High Precision)
            self.residuals = df.iloc[:, 1].values
            self.error = df.iloc[:, 2].values
            
            self.valid = True
            self._clean()
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            
    def _clean(self):
        # Drop NaNs
        mask = ~np.isnan(self.time) & ~np.isnan(self.residuals) & ~np.isnan(self.error)
        self.time = self.time[mask]
        self.residuals = self.residuals[mask]
        self.error = self.error[mask]
        
        # Sort by time
        sort_idx = np.argsort(self.time)
        self.time = self.time[sort_idx]
        self.residuals = self.residuals[sort_idx]
        self.error = self.error[sort_idx]

class SpectralAnalyzer:
    """Core signal processing engine using Lomb-Scargle."""
    
    def __init__(self, time_array, signal_array, error_array=None):
        self.t = time_array
        self.y = signal_array
        self.dy = error_array
        self.frequency = None
        self.power = None
        
    def compute_periodogram(self):
        # Lomb-Scargle with error handling
        if self.dy is not None:
            safe_dy = np.where(self.dy <= 0, 1e-9, self.dy)
            frequency, power = LombScargle(self.t, self.y, safe_dy).autopower()
        else:
            frequency, power = LombScargle(self.t, self.y).autopower()
            
        self.frequency = frequency
        
        # Normalize power for UI slider consistency (0-1 range roughly, or relative peaks)
        # Standard LS power is normalized differently but here we just store raw
        self.power = power
        return frequency, power
        
    def find_peaks_prominence(self, min_prominence=0.1):
        if self.power is None:
            return [], {}
            
        # Normalize power for prominence filter to be intuitive (0.0 - 1.0 relative to max)
        max_power = np.max(self.power) if np.max(self.power) > 0 else 1.0
        normalized_power = self.power / max_power
        
        # Scale prominence threshold to actual power
        abs_prominence = min_prominence * max_power
        
        peaks, properties = find_peaks(self.power, prominence=abs_prominence)
        
        # Add 'norm_prominence' to properties for score calculation
        properties['norm_prominence'] = properties['prominences'] / max_power
        
        return peaks, properties

class GomesDetector:
    """Physics engine looking for specific harmonic signatures."""
    
    GEO_FACTOR = 1.0 + (1.0/12.0)  # ~1.0833
    COUPLING_FACTOR = GOMES_COUPLING
    
    def __init__(self, frequency, power, peaks_indices, peak_props):
        self.freq = frequency
        self.power = power
        self.peaks_indices = peaks_indices
        self.peak_props = peak_props
        self.matches = []
        self.chains = []
        
    def scan_signatures(self, tolerance_percent=1.5):
        self.matches = []
        peak_freqs = self.freq[self.peaks_indices]
        peak_powers = self.power[self.peaks_indices]
        peak_prominences = self.peak_props.get('prominences', np.ones_like(peak_powers))
        
        # O(N^2) scan - Unlimited
        for i in range(len(peak_freqs)):
            for j in range(len(peak_freqs)):
                if i == j: continue
                
                f1 = peak_freqs[i]
                f2 = peak_freqs[j]
                
                ratio = f2 / f1 if f2 > f1 else f1 / f2
                
                match_type = None
                color = None
                
                # Check Geometric Factor
                if self._is_match(ratio, self.GEO_FACTOR, tolerance_percent):
                    match_type = 'Geometric (1.0833)'
                    color = '#00B4D8' # Cyan
                    
                # Check Coupling Factor
                elif self._is_match(ratio, self.COUPLING_FACTOR, tolerance_percent):
                    match_type = f'Coupling ({self.COUPLING_FACTOR:.1f})'
                    color = '#00FF00' # Green

                if match_type:
                    # UPDATED SCORE FORMULA: Favor peaks that stand out (Power * Prominence)
                    # This dramatically weights signals over background noise
                    s1 = peak_powers[i] * peak_prominences[i]
                    s2 = peak_powers[j] * peak_prominences[j]
                    energy_score = s1 * s2
                    
                    self.matches.append({
                        'type': match_type,
                        'f1': f1,
                        'f2': f2,
                        'p1': peak_powers[i],
                        'p2': peak_powers[j],
                        'energy': energy_score,
                        'ratio_detected': ratio,
                        'color': color
                    })
        
        # CHAIN DETECTION (TOPOLOGY)
        self._find_harmonic_chains(peak_freqs, peak_powers, tolerance_percent)
                    
        return self.matches
    
    def _find_harmonic_chains(self, freqs, powers, tol):
        """Find sequences A -> B -> C connected by GEO_FACTOR."""
        self.chains = []
        if len(freqs) < 3: return

        # Build Graph
        G = nx.DiGraph()
        for i, f_start in enumerate(freqs):
            G.add_node(i, freq=f_start, power=powers[i])
            
        for i in range(len(freqs)):
            for j in range(len(freqs)):
                if i == j: continue
                
                # Check A -> B (B = A * 1.0833)
                f_target = freqs[i] * self.GEO_FACTOR
                if self._is_approx(freqs[j], f_target, tol):
                    G.add_edge(i, j)
        
        # Find weakly connected components first
        for component in nx.weakly_connected_components(G):
            if len(component) >= 3:
                subgraph = G.subgraph(component)
                # Find longest path in this subgraph (DAG)
                try:
                    path = nx.dag_longest_path(subgraph)
                    if len(path) >= 3:
                        chain_freqs = [freqs[idx] for idx in path]
                        chain_powers = [powers[idx] for idx in path]
                        
                        # Calculate Chain Energy
                        chain_energy = np.prod(chain_powers) * len(path)
                        
                        self.chains.append({
                            'path_indices': path,
                            'freqs': chain_freqs,
                            'powers': chain_powers,
                            'length': len(path),
                            'energy': chain_energy
                        })
                except:
                    pass 
    
    def _is_match(self, value, target, tol_percent):
        delta = target * (tol_percent / 100.0)
        return (target - delta) <= value <= (target + delta)

    def _is_approx(self, value, target, tol_percent):
        delta = target * (tol_percent / 100.0)
        return (target - delta) <= value <= (target + delta)
        
    def get_total_energy(self):
        # Base pair energy + Chain bonus
        pair_energy = sum(m['energy'] for m in self.matches)
        chain_bonus = sum(c['energy'] for c in self.chains) * 10.0 
        return pair_energy + chain_bonus

class MonteCarloValidator:
    """The Judge."""
    
    @staticmethod
    def run_permutation_test(time_arr, signal_arr, error_arr, original_score, iterations, tolerance, min_prominence):
        scores = []
        progress_bar = st.progress(0)
        
        for i in range(iterations):
            shuffled_signal = np.random.permutation(signal_arr)
            
            # Analyze
            sa = SpectralAnalyzer(time_arr, shuffled_signal, error_arr)
            f, p = sa.compute_periodogram()
            peaks, props = sa.find_peaks_prominence(min_prominence=min_prominence)
            
            # Detect
            gd = GomesDetector(f, p, peaks, props)
            gd.scan_signatures(tolerance_percent=tolerance)
            scores.append(gd.get_total_energy())
            
            progress_bar.progress((i + 1) / iterations)
            
        progress_bar.empty()
        scores = np.array(scores)
        mean_noise = np.mean(scores)
        std_noise = np.std(scores)
        
        if std_noise == 0:
            sigma = 0.0
        else:
            sigma = (original_score - mean_noise) / std_noise
            
        return sigma, mean_noise, scores


# --- MAIN UI LOGIC ---

def scan_local_files():
    files = []
    for root, dirs, filenames in os.walk('.'):
        for f in filenames:
            if f.endswith('.res'):
                full_path = os.path.join(root, f)
                files.append(full_path)
    return files

def main():
    st.title("🌌 Gomes Cosmological Dashboard")
    st.markdown("_Searching for Kerr Universe Signatures in Pulsar Timing Data_")
    
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None

    # Sidebar
    with st.sidebar:
        st.header("1. Data Ingestion")
        local_files = scan_local_files()
        selection = st.selectbox("📡 Select Target Pulsar (Local)", ["-- Select a file --"] + local_files)
        st.markdown("**OR**")
        uploaded_file = st.file_uploader("Manual Upload (.res / .csv)", type=["csv", "res", "txt"])
        
        source = uploaded_file if uploaded_file else (selection if selection != "-- Select a file --" else None)
            
        st.markdown("---")
        st.subheader("2. Analysis Parameters")
        
        tol = st.slider("Tolerance (%)", 0.1, 5.0, 1.5, 0.1)
        prominence = st.slider("Min Peak Prominence (0-1)", 0.0, 1.0, 0.05, 0.01, help="Filter noise peaks. Value is relative to max power.")
        mc_iter = st.number_input("Monte Carlo Iterations", min_value=10, max_value=1000, value=100)
        
        st.markdown("---")
        st.markdown("#### Theory Constants")
        st.code(f"Geo Factor: {GomesDetector.GEO_FACTOR:.4f}")
        st.code(f"Coupling:   {GomesDetector.COUPLING_FACTOR:.1f}")

    # Main Area
    if source is not None:
        data = PulsarData(source)
        
        if data.valid:
            # Stats Raw
            st.caption(f"Loaded {len(data.time)} data points. Duration: {(data.time.max()-data.time.min()):.2f}s")
            
            # --- 0. Raw Data ---
            with st.expander("📊 Raw Data (Time Series)", expanded=False):
                fig_raw = go.Figure()
                fig_raw.add_trace(go.Scatter(x=data.time, y=data.residuals, mode='lines', line=dict(color='#FAFAFA', width=1)))
                fig_raw.update_layout(title="Timing Residuals", template="plotly_dark", height=250)
                st.plotly_chart(fig_raw, use_container_width=True)

            # --- 1. Spectral Analysis ---
            sa = SpectralAnalyzer(data.time, data.residuals, data.error)
            freq, power = sa.compute_periodogram()
            # USE PROMINENCE instead of percentile
            peaks, props = sa.find_peaks_prominence(min_prominence=prominence)
            
            # --- 2. Signature Detection ---
            detector = GomesDetector(freq, power, peaks, props)
            matches = detector.scan_signatures(tolerance_percent=tol)
            total_energy = detector.get_total_energy()
            
            # --- 3. Results ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Matches Found (Total)", len(matches))
            # SHOW CHAINS METRIC
            c2.metric("Harmonic Chains (3+)", len(detector.chains))
            
            sigma_val = 0.0
            if c3.button("Run Sigma Test"):
                with st.spinner(f"Running {mc_iter} Monte Carlo simulations..."):
                    validator = MonteCarloValidator()
                    sigma, noise_mean, _ = validator.run_permutation_test(
                        data.time, data.residuals, data.error, total_energy, mc_iter, tol, prominence
                    )
                    sigma_val = sigma
                    delta_color = "normal" if sigma < 2 else "inverse"
                    c3.metric("Statistical Significance (σ)", f"{sigma:.2f} σ", delta=f"{sigma:.2f}", delta_color=delta_color)
            else:
                c3.metric("Statistical Significance (σ)", "Pending...")

            # --- 4. Plotting (Optimized) ---
            fig = go.Figure()
            
            # Full Spectrum
            fig.add_trace(go.Scatter(x=freq, y=power, mode='lines', name='Power Spectrum', line=dict(color='#212529', width=1), opacity=0.5))
            
            # Peaks
            peak_freqs = freq[peaks]
            peak_powers = power[peaks]
            fig.add_trace(go.Scatter(x=peak_freqs, y=peak_powers, mode='markers', name='Detected Peaks', marker=dict(color='white', size=5, symbol='x')))
            
            # Matches - Display LIMIT to prevent crash
            DISPLAY_LIMIT = 500
            sorted_matches = sorted(matches, key=lambda x: x['energy'], reverse=True)
            display_matches = sorted_matches[:DISPLAY_LIMIT]
            
            if len(matches) > DISPLAY_LIMIT:
                st.warning(f"⚠️ Displaying only top {DISPLAY_LIMIT} matches out of {len(matches)} to preserve browser performance. Statistical score includes ALL matches.")

            # Plot Pairs
            for m in display_matches:
                fig.add_trace(go.Scatter(
                    x=[m['f1'], m['f2']],
                    y=[m['p1'], m['p2']],
                    mode='lines+markers',
                    name=m['type'],
                    line=dict(color=m['color'], width=1.5, dash='dot'),
                    hoverinfo='text',
                    text=f"{m['type']} (Ratio: {m['ratio_detected']:.4f}, E: {m['energy']:.2e})",
                    showlegend=False
                ))
            
            # Plot Chains (GOLD)
            for chain in detector.chains:
                fig.add_trace(go.Scatter(
                    x=chain['freqs'],
                    y=chain['powers'],
                    mode='lines+markers+text',
                    name='HARMONIC CHAIN',
                    line=dict(color='#FFD700', width=3), # Gold
                    marker=dict(color='#FFD700', size=8, symbol='star'),
                    text=[""] * (len(chain['freqs'])-1) + ["CHAIN"],
                    textposition="top center",
                    hoverinfo='text',
                    hovertext=f"Cascade Length: {chain['length']}",
                ))

            # Dummy legend entries for colors
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Geometric (1.0833)', line=dict(color='#00B4D8', dash='dot')))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Coupling (30.6)', line=dict(color='#00FF00', dash='dot')))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Harmonic Chain', line=dict(color='#FFD700', width=3)))
            
            fig.update_layout(title="Lomb-Scargle Periodogram & Detected Harmonic Signatures", xaxis_title="Frequency (Hz)", yaxis_title="Spectral Power", template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Export Section ---
            st.markdown("### 📥 Export Scientific Data")
            
            # Prepare export data
            export_data = []
            
            # Chains
            for i, c in enumerate(detector.chains):
                for j, f in enumerate(c['freqs']):
                    export_data.append({
                        'ID': f"CHAIN_{i}",
                        'Type': 'Harmonic Chain Node',
                        'Frequency': f,
                        'Power': c['powers'][j],
                        'Sigma': sigma_val if sigma_val != 0 else "N/A"
                    })
            
            # Pairs
            for i, m in enumerate(matches):
                export_data.append({
                    'ID': f"MATCH_{i}",
                    'Type': m['type'],
                    'Frequency': m['f1'],
                    'Power': m['p1'],
                    'Sigma': sigma_val if sigma_val != 0 else "N/A"
                })
                
            if export_data:
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Discovery Data (CSV)",
                    data=csv,
                    file_name=f"gomes_discovery_{data.filename}.csv",
                    mime="text/csv"
                )

            with st.expander("Spectrum Details & Target Matches"):
                if matches:
                    st.dataframe(pd.DataFrame(sorted_matches))
                else:
                    st.info("No harmonic signatures detected under current parameters.")
        else:
            pass
    else:
        st.info("👋 Select a Pulsar dataset from the sidebar to begin analysis.")

if __name__ == "__main__":
    from streamlit.web import cli as stcli
    from streamlit import runtime
    import sys
    import os
    
    if not runtime.exists():
        print("🚀 Detected direct execution ('python app.py'). Relaunching via 'streamlit run'...")
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
    
    main()
