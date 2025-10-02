# 🤖 AI Development Tools

> Comprehensive toolkit integrating **Jupyter Notebooks**, **MLflow**, **Weights & Biases**, and **Streamlit** for AI model development and deployment.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![W&B](https://img.shields.io/badge/W%26B-Integration-yellow.svg)](https://wandb.ai/)

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tool Documentation](#tool-documentation)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [MLflow](#mlflow)
  - [Weights & Biases](#weights--biases)
  - [Streamlit Apps](#streamlit-apps)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This repository provides a complete, production-ready toolkit for ML engineers and data scientists to:

- **Experiment** with models using Jupyter Notebooks
- **Track** experiments with MLflow and Weights & Biases
- **Visualize** results with interactive Streamlit dashboards
- **Deploy** models seamlessly with proper version control

## ✨ Features

✅ **Integrated Experiment Tracking**: Unified approach using MLflow and W&B  
✅ **Interactive Dashboards**: Real-time visualization with Streamlit  
✅ **Reproducible Research**: Version control for models, data, and experiments  
✅ **Production Ready**: Best practices for ML deployment  
✅ **Modular Design**: Easy to extend and customize  
✅ **Well Documented**: Comprehensive guides and examples  

## 📜 Project Structure

```
ai-development-tools/
│
├── jupyter_notebooks/          # Jupyter notebook demos and templates
│   └── demo_notebook.ipynb      # Sample notebook with MLflow & W&B integration
│
├── mlflow_demo/               # MLflow experiment tracking demos
│   └── mlflow_tracking.py       # Example MLflow tracking implementation
│
├── wandb_tracking/            # Weights & Biases integration
│   └── wandb_integration.py     # W&B tracking and sweeps
│
├── streamlit_apps/            # Streamlit dashboard applications
│   └── model_dashboard.py       # Interactive model performance dashboard
│
├── .env.example               # Environment variables template
├── .gitignore                 # Python gitignore
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/rahit91890/ai-development-tools.git
cd ai-development-tools
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Core dependencies
pip install jupyter mlflow wandb streamlit

# ML libraries
pip install numpy pandas scikit-learn

# Visualization
pip install plotly matplotlib seaborn

# Or install all at once:
pip install jupyter mlflow wandb streamlit numpy pandas scikit-learn plotly matplotlib seaborn
```

4. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## 🚀 Quick Start

### 1. Start Jupyter Notebook

```bash
jupyter notebook
# Navigate to jupyter_notebooks/demo_notebook.ipynb
```

### 2. Run MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --port 5000

# In another terminal, run the demo
python mlflow_demo/mlflow_tracking.py
```

### 3. Use Weights & Biases

```bash
# Login to W&B (first time only)
wandb login

# Run W&B integration demo
python wandb_tracking/wandb_integration.py
```

### 4. Launch Streamlit Dashboard

```bash
streamlit run streamlit_apps/model_dashboard.py
```

## 📊 Tool Documentation

### Jupyter Notebooks

**Location**: `jupyter_notebooks/`

Interactive Python notebooks for:
- Exploratory data analysis
- Model prototyping
- Experiment documentation
- Integration with MLflow and W&B

**Key Features**:
- Pre-configured for ML workflows
- Integrated experiment tracking
- Visualization templates

### MLflow

**Location**: `mlflow_demo/`

MLflow provides:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version control for models
- **Model Deployment**: Serve models as APIs

**Usage**:
```python
import mlflow

mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("param_name", param_value)
    mlflow.log_metric("metric_name", metric_value)
    mlflow.sklearn.log_model(model, "model")
```

**View Results**:
```bash
mlflow ui
# Open http://localhost:5000
```

### Weights & Biases

**Location**: `wandb_tracking/`

W&B offers:
- **Experiment Tracking**: Advanced visualization and comparison
- **Hyperparameter Sweeps**: Automated hyperparameter optimization
- **Artifacts**: Store and version models, datasets
- **Reports**: Share results with stakeholders

**Usage**:
```python
import wandb

wandb.init(project="my-project", config=config)

# Log metrics
wandb.log({"accuracy": 0.95, "loss": 0.05})

# Log artifacts
wandb.save("model.pkl")
```

**Hyperparameter Sweeps**:
See `wandb_tracking/wandb_integration.py` for sweep configuration examples.

### Streamlit Apps

**Location**: `streamlit_apps/`

Interactive web applications for:
- Real-time model monitoring
- Experiment comparison
- Performance visualization
- Stakeholder demos

**Features**:
- 📊 Interactive charts and metrics
- 🔍 Experiment filtering and comparison
- 💾 Model performance tracking
- 📡 Live data updates

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=my-experiment

# Weights & Biases
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=my-project

# Streamlit
STREAMLIT_SERVER_PORT=8501

# Model Storage
MODEL_SAVE_PATH=./models
DATA_PATH=./data
```

### MLflow Configuration

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")
```

### W&B Configuration

Set your API key:
```bash
export WANDB_API_KEY=your_key_here
# Or use: wandb login
```

## 💻 Usage Examples

### Example 1: Train with MLflow Tracking

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("rf-classifier")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Example 2: W&B Hyperparameter Sweep

```python
import wandb

# Define sweep configuration
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'min': 0.001, 'max': 0.1},
        'batch_size': {'values': [16, 32, 64]}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, train_function, count=10)
```

### Example 3: Streamlit Dashboard

```python
import streamlit as st
import pandas as pd

st.title("Model Performance Dashboard")

# Load metrics
metrics_df = pd.read_csv("metrics.csv")

# Display metrics
st.line_chart(metrics_df[["accuracy", "loss"]])
st.metric("Best Accuracy", metrics_df["accuracy"].max())
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Contact

**Rahit Biswas**
- GitHub: [@rahit91890](https://github.com/rahit91890)
- Email: r.codaphics@gmail.com
- Website: [codaphics.com](https://codaphics.com)

## 🌟 Acknowledgments

- [MLflow](https://mlflow.org/) - Open source platform for the ML lifecycle
- [Weights & Biases](https://wandb.ai/) - Developer tools for ML
- [Streamlit](https://streamlit.io/) - Fastest way to build data apps
- [Jupyter](https://jupyter.org/) - Interactive computing

---

⭐️ **If you find this project helpful, please give it a star!** ⭐️
