#!/bin/bash
set -e

echo "=== CADENCE Reproducibility Setup ==="

# 1. Install PlatformIO CLI Globally
echo "[+] Installing PlatformIO CLI globally..."
python3 -m pip install -U platformio

# 2. Setup Python Virtual Environment for ML
echo "[+] Creating Python virtual environment for Machine Learning pipeline..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Install ML Dependencies
echo "[+] Installing pip dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify Installations
echo "[+] Verifying installations..."
pio --version
python -c "import tensorflow; print('TensorFlow verified')"
python -c "import parselmouth; print('Praat-Parselmouth verified')"

echo "=== Setup Complete ==="
echo "To build the C++ firmware: pio run -e phase_c"
echo "To train the models: source .venv/bin/activate && python data_collection/train.py"
