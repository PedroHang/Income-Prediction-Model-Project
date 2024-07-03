# Income Prediction Project

Welcome to the Income Prediction Project! This guide will help you set up the project environment and install all the necessary dependencies. For the best experience, we recommend using Anaconda Navigator.

**Notes:**
- The project's `.ipynb` (Jupyter Notebook) is written in Portuguese (pt-BR).
- The project's Streamlit file, `income_prediction_streamlit.py`, is crucial for understanding the project.

## Quick Setup Instructions

### Step 1: Install Anaconda Navigator

If you haven't already, download and install [Anaconda Navigator](https://www.anaconda.com/products/distribution). Anaconda simplifies package management and deployment.

### Step 2: Create a New Environment

Open Anaconda Navigator and create a new environment for this project. This helps to keep dependencies organized and prevents conflicts with other projects.

1. Open Anaconda Navigator.
2. Click on the **Environments** tab.
3. Click the **Create** button.
4. Name your environment (e.g., `income-prediction-env`).
5. Choose the Python version you need (e.g., Python 3.8).
6. Click **Create** to create the environment.

### Step 3: Install Project Dependencies

Activate your new environment and open a terminal. Run the following command to install all required dependencies:

```bash
pip install patsy statsmodels ydata-profiling pandas matplotlib seaborn numpy streamlit scikit-learn plotly
```

This command installs the following packages:

- **patsy**: A Python library for describing statistical models and building design matrices.
- **statsmodels**: Provides classes and functions for the estimation of many different statistical models.
- **ydata-profiling**: A tool for generating profile reports from a pandas DataFrame.
- **pandas**: A powerful data manipulation and analysis library.
- **matplotlib**: A plotting library for creating static, animated, and interactive visualizations.
- **seaborn**: A Python visualization library based on matplotlib, providing a high-level interface for drawing attractive statistical graphics.
- **numpy**: A fundamental package for array computing with Python.
- **streamlit**: A fast way to build and share data apps.
- **scikit-learn**: A machine learning library for Python.

### Step 4: Run the Project

After installing the dependencies, you are ready to run the project. The project's Streamlit file, `income_prediction_streamlit.py`, is crucial for understanding the project.

#### Running the Streamlit File

1. Ensure you are in the correct directory where `income_prediction_streamlit.py` is located.
2. Open a terminal.
3. Activate your environment if it is not already activated:
   ```bash
   conda activate income-prediction-env

4. Run the Streamlit file using the following command:
```bash
   streamlit run income_prediction_streamlit.py
```
5. This will start the Streamlit server, and you will see output in the terminal indicating that the app is running. You can view the app in your web browser by navigating to the URL provided, typically `http://localhost:8501`.

---

Thank you for your time!!




