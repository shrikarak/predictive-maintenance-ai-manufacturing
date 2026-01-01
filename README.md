# AI for Predictive Maintenance in Manufacturing

Copyright (c) 2026 Shrikara Kaudambady. All rights reserved.

## 1. Introduction

In the manufacturing and aerospace industries, unplanned equipment failure is a major source of operational disruption and financial loss. Predictive Maintenance is an AI-driven strategy that shifts maintenance from a reactive ("fix it when it breaks") or scheduled ("fix it every N months") approach to a proactive, intelligent one.

This project provides a Jupyter Notebook that implements a deep learning model to predict the **Remaining Useful Life (RUL)** of a machine based on its sensor data. By forecasting when a component is likely to fail, maintenance can be scheduled just in time, maximizing component lifespan and minimizing downtime.

For this demonstration, we use the well-known **NASA Turbofan Engine Degradation Simulation Dataset**.

## 2. The Solution Explained: Time-Series Forecasting with LSTMs

The core of this solution is to treat the RUL prediction as a **sequence-to-value regression problem**. We use the recent history of a machine's sensor readings to predict a single numerical value: the number of operational cycles it has left before failure.

### 2.1 The Dataset

The NASA dataset simulates time-series data from a fleet of turbofan engines. Each engine starts in a healthy state and runs for a variable number of cycles until it experiences a failure. The data includes:
*   An engine ID.
*   The current operational cycle number.
*   Readings from 21 different sensors (e.g., temperature, pressure, fan speed).

### 2.2 The AI Model: Long Short-Term Memory (LSTM) Network

Because equipment degradation is a process that occurs over time, we need a model that can understand temporal patterns in sequential data. An **LSTM (Long Short-Term Memory)** network, a type of Recurrent Neural Network (RNN), is perfectly suited for this task.

**How it Works:**
1.  **Data Preprocessing:** For each engine in the training set, we first calculate the RUL at every cycle. The sensor data is then scaled to a uniform range.
2.  **Sequencing:** We transform the time-series data into "windows." For example, we create input sequences containing the last 50 cycles of sensor data and a corresponding target label, which is the RUL at the end of that 50-cycle window.
3.  **Learning Degradation Patterns:** The LSTM network processes these sequences. Its internal "memory cells" allow it to learn the relationships between changes in sensor readings over time and the corresponding decrease in RUL. It learns the signature of a machine that is degrading and approaching failure.

### 2.3 Evaluation

The model is evaluated on a separate test set where the final RUL values are known but are not used during training. We measure the model's performance using **Root Mean Squared Error (RMSE)**, which gives us an idea of how many cycles our RUL predictions are off by, on average. We also visualize the results with a scatter plot of `Predicted RUL` vs. `Actual RUL`.

## 3. How to Use the Notebook

### 3.1. Prerequisites

This project uses the TensorFlow deep learning library and standard data science packages.

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### 3.2. Running the Notebook

1.  Clone this repository:
    ```bash
    git clone https://github.com/shrikarak/predictive-maintenance-ai-manufacturing.git
    cd predictive-maintenance-ai-manufacturing
    ```
2.  Start the Jupyter server:
    ```bash
    jupyter notebook
    ```
3.  Open `predictive_maintenance_lstm.ipynb` and run the cells sequentially. The notebook will automatically download the NASA dataset, preprocess the data, build and train the LSTM model, and evaluate its performance.

## 4. Deployment and Real-World Application

This notebook provides a complete template for building a predictive maintenance system.

1.  **Save and Deploy the Model:** The trained LSTM model can be saved and deployed as an API service.
2.  **Real-Time Monitoring:** In a factory setting, this API would receive live streams of sensor data from machinery in operation.
3.  **Continuous Prediction:** For each machine, the system would continuously feed the latest window of sensor data to the model to get an updated RUL prediction.
4.  **Automated Alerting:** When a machine's predicted RUL drops below a pre-defined safety threshold (e.g., 20 cycles), an automatic work order can be generated in a maintenance management system, or an alert can be sent to the maintenance team to schedule a service, thus preventing a catastrophic failure.
