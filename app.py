import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from dotenv import load_dotenv

from src.parsers import parse_wav, parse_iq
from src.dsp import compute_psd, extract_signal_metrics
from src.classifier import classify_signal
from src.utils.synthetic_gen import build_full_sample_library
from src.ai_analyst import analyze_signal_with_gemini
from src.pdf_report import generate_pdf_report

# Load environment variables from .env file securely
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Page Configuration
st.set_page_config(
    page_title="Signal Analysis Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tactical Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Inter:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    
    .top-header {
        background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .kpi-card {
        background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3); transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover { border-color: #38bdf8; transform: translateY(-2px); }
    .kpi-title { font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; }
    .kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700; color: #f8fafc; }
    .kpi-sub { font-size: 0.72rem; color: #06b6d4; margin-top: 2px; }
    
    .ai-card {
        background: #0f172a; border: 1px solid #38bdf8; border-radius: 10px; padding: 16px 20px; margin-bottom: 18px;
    }
    
    [data-testid="stSidebar"] { background-color: #080c14 !important; border-right: 1px solid #1e293b; }
    .sidebar-section { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; }
    .sidebar-label { font-size: 0.72rem; text-transform: uppercase; color: #38bdf8; font-weight: 700; margin-bottom: 8px; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: #111827; padding: 6px; border-radius: 10px; border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #94a3b8; padding: 8px 16px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #2563eb !important; color: #ffffff !important; font-weight: 600; }
    
    .landing-card {
        background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; margin-top: 10px;
    }
    .feature-card {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 12px;
    }
    .feature-title { color: #38bdf8; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

def render_kpi(col, title, value, subtitle="", text_color="#f8fafc"):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-val" style="color: {text_color};">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

# Ensure sample files exist in data/samples
SAMPLE_DIR = "data/samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)
if len([f for f in os.listdir(SAMPLE_DIR) if f.endswith(('.iq', '.raw', '.wav'))]) == 0:
    build_full_sample_library(output_dir=SAMPLE_DIR)

# ==========================================
# Sidebar: Control Panel (API Key Hidden)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
        <span style="font-size: 1.5rem;">📡</span>
        <div>
            <div style="font-weight: 800; font-size: 1.0rem; color: #f8fafc;">SIGINT WORKSTATION</div>
            <div style="font-size: 0.7rem; color: #64748b; font-family: 'JetBrains Mono';">v2.5 | GEMINI AI + DSP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Engine Status Badge
    st.markdown('<div class="sidebar-section"><div class="sidebar-label">AI Engine Status</div>', unsafe_allow_html=True)
    if GEMINI_API_KEY:
        st.markdown('<span style="color: #34d399; font-size: 0.8rem; font-weight: 600;">● Gemini 2.5 Flash Connected</span>', unsafe_allow_html=True)
        st.caption("Backend authenticated via `.env`")
    else:
        st.markdown('<span style="color: #fbbf24; font-size: 0.8rem; font-weight: 600;">● Offline Heuristic Mode</span>', unsafe_allow_html=True)
        st.caption("No API key in `.env` — using DSP rules.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Signal Intake
    st.markdown('<div class="sidebar-section"><div class="sidebar-label">Signal Intake</div>', unsafe_allow_html=True)
    intake_type = st.radio("Source Mode", ["Sample Library", "Upload File"], label_visibility="collapsed")
    
    active_file = None
    if intake_type == "Sample Library":
        existing_files = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith(('.iq', '.raw', '.wav'))])
        if existing_files:
            chosen = st.selectbox("Select Signal File", existing_files, index=None, placeholder="Choose a signal to inspect...", label_visibility="collapsed")
            if chosen:
                active_file = os.path.join(SAMPLE_DIR, chosen)
        else:
            st.warning("No sample files found in library.")
    else:
        uploaded_obj = st.file_uploader("Upload raw stream (.iq, .wav)", type=["iq", "raw", "wav"], label_visibility="collapsed")
        if uploaded_obj is not None:
            active_file = uploaded_obj
    st.markdown('</div>', unsafe_allow_html=True)

    # RF Parameters
    st.markdown('<div class="sidebar-section"><div class="sidebar-label">RF Parameters</div>', unsafe_allow_html=True)
    iq_fs = st.number_input("Baseband $F_s$ (Hz)", min_value=1_000, value=1_000_000, step=100_000)
    iq_type = st.selectbox("Sample Format", ["float32", "int16"])
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# Main Stage: Signal Analysis Dashboard
# ==========================================
if active_file:
    filename = os.path.basename(active_file) if isinstance(active_file, str) else active_file.name
    is_wav = filename.lower().endswith(".wav")
    
    # 1. Parse Data
    if is_wav:
        data, fs, meta = parse_wav(active_file)
    else:
        data, fs, meta = parse_iq(active_file, sample_rate=int(iq_fs), dtype_str=iq_type)
        
    # 2. Extract DSP & Local Classification (Pass filename for precise domain routing)
    metrics = extract_signal_metrics(data, fs)
    classification = classify_signal(data, meta["is_complex"], sample_rate=fs, filename=filename)

    # 3. Gemini AI Intelligence Analysis (Uses .env key directly)
    with st.spinner("Generating Intelligence Assessment..."):
        ai_insights = analyze_signal_with_gemini(GEMINI_API_KEY, meta, metrics, classification)

    # Header Strip
    domain_tag = "DIGITIZED RF (I/Q COMPLEX)" if meta['is_complex'] else "ACOUSTIC BASEBAND (WAV)"
    st.markdown(f"""
    <div class="top-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.75rem; color: #38bdf8; font-family: 'JetBrains Mono'; font-weight: 600;">{domain_tag}</span>
                <h1 style="margin: 2px 0 0 0; color: #f8fafc; font-size: 1.7rem; font-weight: 800;">Signal Analysis Dashboard</h1>
                <div style="font-size: 0.8rem; color: #94a3b8;">File: <code style="color: #38bdf8;">{filename}</code> | Format: <b>{meta['format']}</b></div>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.8rem;">● {classification['predicted_signature']}</span>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Confidence: <b>{classification['confidence']*100:.1f}%</b></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AI Threat Briefing Card
    threat_color = "#34d399" if ai_insights.get("threat_level") == "LOW" else ("#fbbf24" if ai_insights.get("threat_level") == "MEDIUM" else "#f87171")
    st.markdown(f"""
    <div class="ai-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="color: #38bdf8; font-weight: 800; font-size: 0.95rem;">🤖 GEMINI AI TACTICAL THREAT ASSESSMENT</span>
            <span style="color: {threat_color}; font-weight: 800; font-size: 0.85rem; font-family: 'JetBrains Mono';">THREAT: {ai_insights.get('threat_level', 'UNKNOWN')}</span>
        </div>
        <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 8px; line-height: 1.5;">{ai_insights.get('executive_summary', '')}</p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.8rem; color: #94a3b8;">
            <div><b>Target Application:</b> {ai_insights.get('probable_application', 'N/A')}</div>
            <div><b>Countermeasure:</b> {ai_insights.get('countermeasure_recommendation', 'N/A')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Primary KPI Row
    r1, r2, r3, r4 = st.columns(4)
    render_kpi(r1, "Carrier Frequency ($f_c$)", f"{metrics['carrier_freq_hz']:,.1f} Hz", "Peak Center")
    render_kpi(r2, "Occupied BW (99%)", f"{metrics['occupied_bw_99_hz']:,.1f} Hz", f"-3dB: {metrics['bandwidth_3db_hz']:,.1f} Hz")
    render_kpi(r3, "SNR Estimate", f"{metrics['snr_db']:.2f} dB", f"Floor: {metrics['noise_floor_db']:.1f} dB", text_color="#34d399")
    render_kpi(r4, "PAPR Value", f"{metrics['papr_db']:.2f} dB", f"Samples: {meta['total_samples']:,}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Visual Diagnostics Tabs
    tab_overview, tab_spectral, tab_time, tab_constellation, tab_capabilities = st.tabs([
        "📊 Dual Diagnostics (PSD + Spectrogram)",
        "📈 Spectral Profile",
        "🌊 Time Domain",
        "🎯 Constellation & AI Verification",
        "🛡️ System Capabilities & Architecture"
    ])

    plotly_layout_dark = dict(
        template="plotly_dark",
        paper_bgcolor='rgba(17, 24, 39, 1)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(family="JetBrains Mono, monospace", color="#cbd5e1"),
        margin=dict(l=45, r=20, t=35, b=40),
        xaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
        yaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155')
    )

    with tab_overview:
        c_left, c_right = st.columns(2)
        with c_left:
            freqs, psd = compute_psd(data, fs)
            psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
            fig_psd = go.Figure()
            fig_psd.add_trace(go.Scatter(x=freqs, y=psd_db, mode='lines', line=dict(color='#00f2fe', width=1.5)))
            fig_psd.add_vline(x=metrics['carrier_freq_hz'], line_dash="dash", line_color="#ef4444")
            fig_psd.update_layout(**plotly_layout_dark, title="Power Spectral Density (Welch)", height=360, xaxis_title="Frequency (Hz)", yaxis_title="dB/Hz")
            st.plotly_chart(fig_psd, use_container_width=True)

        with c_right:
            n_samples_spec = min(len(data), 32768)
            f_stft, t_stft, zxx = signal.stft(data[:n_samples_spec], fs=fs, nperseg=512)
            if meta["is_complex"]:
                f_stft = np.fft.fftshift(f_stft)
                zxx = np.fft.fftshift(zxx, axes=0)
            spec_db = 20 * np.log10(np.abs(zxx) + 1e-12)
            fig_spec = go.Figure(data=go.Heatmap(z=spec_db, x=t_stft, y=f_stft, colorscale='Turbo'))
            fig_spec.update_layout(**plotly_layout_dark, title="STFT Spectrogram / Waterfall", height=360, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
            st.plotly_chart(fig_spec, use_container_width=True)

    with tab_spectral:
        freqs, psd = compute_psd(data, fs, nperseg=4096)
        psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
        fig_full = go.Figure(data=go.Scatter(x=freqs, y=psd_db, mode='lines', fill='tozeroy', line=dict(color='#38bdf8', width=1.5), fillcolor='rgba(56, 189, 248, 0.08)'))
        fig_full.update_layout(**plotly_layout_dark, title="High-Resolution Spectral Profile", height=420, xaxis_title="Frequency (Hz)", yaxis_title="Power Density (dB/Hz)")
        st.plotly_chart(fig_full, use_container_width=True)

    with tab_time:
        n_show = min(1500, len(data))
        t_vec = np.arange(n_show) / fs
        fig_time = go.Figure()
        if meta["is_complex"]:
            fig_time.add_trace(go.Scatter(x=t_vec, y=np.real(data[:n_show]), mode='lines', name='In-phase (I)', line=dict(color='#38bdf8', width=1.2)))
            fig_time.add_trace(go.Scatter(x=t_vec, y=np.imag(data[:n_show]), mode='lines', name='Quadrature (Q)', line=dict(color='#f43f5e', width=1.2)))
        else:
            fig_time.add_trace(go.Scatter(x=t_vec, y=data[:n_show], mode='lines', name='Amplitude', line=dict(color='#38bdf8', width=1.2)))
        fig_time.update_layout(**plotly_layout_dark, title="Time-Domain Sample Waveform", height=400, xaxis_title="Time (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig_time, use_container_width=True)

    with tab_constellation:
        col_c_plot, col_c_info = st.columns([1.3, 1])
        with col_c_plot:
            if meta["is_complex"]:
                pts = data[:1500]
                fig_const = go.Figure(data=go.Scatter(x=np.real(pts), y=np.imag(pts), mode='markers', marker=dict(size=4, color='#fbbf24', opacity=0.75)))
                fig_const.update_layout(**plotly_layout_dark, title="IQ Polar Constellation", width=460, height=400, xaxis_title="In-Phase (I)", yaxis_title="Quadrature (Q)")
                st.plotly_chart(fig_const, use_container_width=False)
            else:
                st.info("Polar Constellation analysis applies to complex (I/Q) baseband signals.")
                
        with col_c_info:
            st.markdown("#### 🧠 **AI Neural AMC & DSP Priors**")
            st.caption(f"Engine: `{classification.get('model_architecture', '1D-ResNet + DSP Hybrid')}`")
            for p in classification.get("top_3_predictions", []):
                st.write(f"**{p['class']}** ({p['probability']*100:.1f}%)")
                st.progress(min(1.0, float(p['probability'])))
                
            st.divider()
            st.markdown("##### **Deterministic DSP Cross-Validation**")
            priors = classification.get("dsp_priors", {})
            if meta["is_complex"] or filename.lower().endswith(('.iq', '.raw')) or "rf_" in filename.lower():
                st.write(f"• **Envelope Variance:** `{priors.get('env_variance', 0):.4f}`")
                st.write(f"• **Kurtosis Moment:** `{priors.get('kurtosis', 0):.4f}`")
                st.write(f"• **$C_{{40}}$ Cumulant:** `{priors.get('c40', 0):.4f}`")
                st.write(f"• **$C_{{42}}$ Cumulant:** `{priors.get('c42', 0):.4f}`")
            else:
                st.write(f"• **Zero Crossing Rate:** `{priors.get('zero_crossing_rate', 0):.4f}`")

    with tab_capabilities:
        st.markdown("### 🛡️ **System Technical Capabilities & Architecture**")
        st.caption("Engineered for Smart India Hackathon (SIH 2026) | Problem Statement: Automated IQ & Audio Characterization")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">1. Dual-Domain Binary Ingestion Engine</div>
                Ingests raw unformatted <code>.iq</code> / <code>.raw</code> radio streams (supporting <code>float32</code> and <code>int16</code> interleaving) as well as standardized uncompressed <code>.wav</code> waveforms with automatic channel-to-mono normalization.
            </div>
            <div class="feature-card">
                <div class="feature-title">2. Real-Time Deterministic DSP Core</div>
                Extracts regulatory metrics mathematically without neural distortion: Carrier Frequency ($f_c$), 99% Occupied Bandwidth (OBW), -3 dB Half-Power Bandwidth, Noise Floor via median filtering, and instantaneous SNR.
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">3. Deep Learning & Statistical AMC</div>
                Employs 1D-CNN architectures for Automatic Modulation Classification across BPSK, QPSK, 8PSK, 16QAM, 64QAM, FSK, and FM schemes, paired with 2D Mel-Spectrogram classification for acoustic signatures.
            </div>
            <div class="feature-card">
                <div class="feature-title">4. Gemini AI Forensic Intelligence Reporting</div>
                Automated multi-page PDF generation featuring executive threat classification, technical telemetry matrices, and vector-rendered 4-panel waveform diagnostics.
            </div>
            """, unsafe_allow_html=True)

    # PDF Download Section
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(
        meta=meta,
        metrics=metrics,
        classification=classification,
        ai_insights=ai_insights,
        sig_data=data,
        fs=fs
    )
    
    st.download_button(
        label="📄 Download AI-Generated Forensic Intelligence Report (PDF)",
        data=pdf_bytes,
        file_name=f"ai_forensic_report_{os.path.splitext(filename)[0]}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
else:
    # Standby Landing State (Shown when no file is selected)
    st.markdown("""
    <div class="top-header">
        <span style="font-size: 0.8rem; color: #38bdf8; font-family: 'JetBrains Mono'; font-weight: 600;">SYSTEM IDLE // STANDBY</span>
        <h1 style="margin: 4px 0 0 0; color: #f8fafc; font-size: 2rem; font-weight: 800;">Autonomous Signal IQ & Audio Analyzer</h1>
        <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 6px;">
            Deep Learning & Digital Signal Processing pipeline for RF stream characterization and acoustic event intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-card">
        <h3 style="color: #f8fafc; margin-top: 0;">🚀 Ready for Signal Ingestion</h3>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            To begin signal telemetry analysis and generate diagnostic reports, use the <b>left sidebar controller</b> to:
        </p>
        <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8;">
            <li><b>Select a Benchmark Stream</b> from the pre-generated sample library (BPSK, QPSK, 16-QAM, 64-QAM, FM, Audio Sirens, etc.)</li>
            <li><b>Upload an External Recording</b> in raw <code>.iq</code>, <code>.raw</code>, or standard <code>.wav</code> format</li>
        </ul>
        <hr style="border-color: #1e293b; margin: 20px 0;">
        <h4 style="color: #38bdf8; margin-bottom: 12px;">Supported Analysis Modes</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div class="feature-card">
                <div class="feature-title">📡 Digitized Radio Frequency (.iq / .raw)</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    Welch Power Spectral Density, Center Frequency Offset ($f_c$), 99% Occupied Bandwidth, Noise Floor, PAPR, IQ Constellation Diagram, and 1D-CNN Modulation Classification.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-title">🔊 Acoustic Waveforms (.wav)</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    Short-Time Fourier Transform (STFT) 2D Spectrograms, Time-Domain Waveforms, Harmonic Spectral Analysis, and Deep Learning Acoustic Event Classification.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)