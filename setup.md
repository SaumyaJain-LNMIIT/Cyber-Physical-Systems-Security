# CPS Security Project - Setup Guide

Welcome to the CPS Security Project! This repository contains a complete pipeline for a Federated Intrusion Detection System with Adversarial Robustness and Explainable AI (XAI), built for the SWaT (Secure Water Treatment) dataset.

## 🚀 Quick Setup Instructions

### 1. Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment so you don't conflict with other Python projects on your computer.
Open your terminal and run:

```bash
# Navigate to the project directory
cd cps_security_project

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
All the required modules have been captured in the `requirements.txt` file. Install them by running:

```bash
pip install -r requirements.txt
```

*(Note: Depending on your system, you may see some warnings during installation. As long as it finishes successfully, you are good to go!)*

### 4. Running the Project
The project consists of two main components: a Backend API (FastAPI) and a Frontend Dashboard (Streamlit).

**Step A: Start the Backend API**
Open a terminal, activate your virtual environment, and run:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*Leave this terminal running in the background.*

**Step B: Start the Frontend Dashboard**
Open a **new** terminal, activate the virtual environment again, and run:
```bash
python -m streamlit run dashboard/app.py
```

This will automatically open the dashboard in your default web browser at `http://localhost:8501`.

---

## 📁 Included in this Zip
- **Source Code**: All the necessary python modules (`api`, `dashboard`, `models`, `preprocessing`, etc.).
- **Pre-trained Weights**: The `experiments/results` folder is included. This contains the `colab_trained_model.pt` (the heavily trained neural network weights), the `colab_scaler.pkl` (for data normalization), and `experiment_results.json` (which holds all the performance, robustness, and SHAP metrics). 
- **Dependencies List**: `requirements.txt` is included so you can install the exact modules needed.

*(Note: The heavy virtual environment folder and cache files were excluded from this zip to keep it lightweight, as you will generate your own using Step 2 & 3).*

Enjoy exploring the project!
