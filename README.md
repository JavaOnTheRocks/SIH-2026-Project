# 📡 Autonomous Signal IQ & Audio Analyzer

An autonomous Signal Intelligence (SIGINT) and Electronic Warfare (EW) characterization workstation designed for raw RF (`.iq`, `.raw`) and acoustic (`.wav`) streams. Bridges deterministic Digital Signal Processing (DSP) with Deep Learning AMC and Google Gemini AI threat analysis.

---

## 📋 Prerequisites

Ensure **Python 3.10 or higher** and **Git** are installed on your machine before proceeding.

Verify your installation:

```bash
python --version
git --version

```

> If Python is not installed, download it from [python.org](https://www.python.org/downloads/) (make sure to check **"Add Python to PATH"** during installation on Windows).

---

## 🚀 Quickstart & Installation

Run the following commands in your terminal to set up and start the application.

### 1. Clone the Repository

```bash
git clone https://github.com/JavaOnTheRocks/Signal-Analysis-Project.git
cd Signal-Analysis-Project

```

---

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell / Command Prompt):**

```bash
python -m venv venv
.\venv\Scripts\activate

```

**On Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate

```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root to enable Gemini AI threat intelligence:

**On Windows (PowerShell):**

```powershell
New-Item -Path .env -ItemType File -Value "GEMINI_API_KEY=your_actual_gemini_api_key_here"

```

**On Linux / macOS:**

```bash
echo "GEMINI_API_KEY=your_actual_gemini_api_key_here" > .env

```

*(Optional: If you do not provide an API key, the system automatically falls back to offline heuristic analysis.)*

---

### 5. Launch the Dashboard

```bash
streamlit run app.py

```

Open your browser and navigate to:

```text
http://localhost:8501

```

---

## 🏗️ System Architecture & Signal Flow

```text
               ┌─────────────────────────────────────────────────────┐
               │         SIGNAL INGESTION & PARSING ENGINE           │
               │  • Raw .iq / .raw (float32, int16 complex streams)  │
               │  • Audio .wav (Mono normalized acoustic streams)    │
               └──────────────────────────┬──────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │  DETERMINISTIC DSP CORE   │               │   DEEP LEARNING ENGINE    │
      │  • Welch PSD & Spectral   │               │  • 1D-CNN RF AMC          │
      │  • 99% OBW & -3dB BW      │               │  • 2D Mel-Spectrogram CNN │
      │  • SNR & Noise Floor      │               │  • Statistical Cumulants  │
      │  • PAPR Calculation       │               │    (C40, C42, Kurtosis)   │
      └─────────────┬─────────────┘               └─────────────┬─────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                       ┌─────────────────────────────────────┐
                       │      GEMINI 2.5 FLASH AI ENGINE     │
                       │  • Protocol & Standard Matching     │
                       │  • Operational Context Assessment   │
                       │  • Threat Level & Countermeasures   │
                       └──────────────────┬──────────────────┘
                                          ▼
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│     STREAMLIT TACTICAL DASHBOARD      │   │     FORENSIC PDF REPORT GENERATOR     │
│  • Dual Diagnostics (PSD + STFT)      │   │  • Multi-Page Executive Intel Report  │
│  • IQ Polar Constellation Diagrams    │   │  • High-Res Waveform Matplotlib Plots │
│  • AI Threat Triage Card              │   │  • Structured DSP Telemetry Tables    │
└───────────────────────────────────────┘   └───────────────────────────────────────┘

```

---

## 📁 Project Directory Structure

```text
signal_analyzer/
│
├── data/
│   ├── samples/                # Benchmark dataset files (.iq, .raw, .wav)
│   └── temp/                   # Temporary cache directory
│
├── src/
│   ├── __init__.py
│   ├── parsers.py              # Ingestion engine for raw RF and audio streams
│   ├── dsp.py                  # Deterministic mathematical DSP calculation routines
│   ├── classifier.py           # PyTorch 1D/2D CNN classification & cumulant validation
│   ├── ai_analyst.py           # Google Gemini 2.5 Flash intelligence & threat engine
│   ├── pdf_report.py           # Multi-section forensic PDF generator (ReportLab)
│   └── utils/
│       ├── __init__.py
│       └── synthetic_gen.py    # Multi-modulation synthetic benchmark generator
│
├── .env                        # Local environment variables (GEMINI_API_KEY)
├── .gitignore
├── app.py                      # Streamlit tactical operations dashboard
├── requirements.txt            # Project dependencies
└── README.md                   # Setup documentation

```

---

## 🎥 Walkthrough & Functional Demo

> *Functional demonstration video showcase will be linked here.*
