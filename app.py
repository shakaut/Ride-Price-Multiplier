import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class MyRandomForestRegressor:
    def __init__(self, n_estimators=100, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(
                max_features=self.max_features,
                random_state=(self.random_state + i) if self.random_state is not None else None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_depth=None
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(all_preds, axis=0)
    

# Load trained model and preprocessor
rf = joblib.load("rf_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Set up the UI
st.subheader("Ride Price Multiplier Prediction App")
st.write('')
st.markdown('Final Fare = Base Fare x Price Multiplier')

st.sidebar.header("Input Features")

# Define inputs
area_count = 30  # set this to your actual max area_id
area_id = st.sidebar.slider("Area ID", 1, area_count, 5)
distance_km = st.sidebar.number_input("Distance (km)", 0.5, 20.0, 5.0, step=0.1)
active_drivers = st.sidebar.number_input("Active Drivers", 1, 100, 10, step=1)
pending_requests = st.sidebar.number_input("Pending Requests", 0, 100, 5, step=1)
traffic = st.sidebar.selectbox("Traffic", ['low', 'medium', 'high'])
weather = st.sidebar.selectbox("Weather", ['clear', 'rainy'])
hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
is_event = st.sidebar.selectbox("Event?", [0, 1])

# Convert input into DataFrame
input_df = pd.DataFrame([{
    'area_id': area_id,
    'distance_km': distance_km,
    'active_drivers': active_drivers,
    'pending_requests': pending_requests,
    'traffic': traffic,
    'weather': weather,
    'hour': hour,
    'day_of_week': day_of_week,
    'is_event': is_event
}])

# Preprocess input
input_processed = preprocessor.transform(input_df)

# Predict
prediction = rf.predict(input_processed)[0]

st.metric(label="Multiplier", value=round(prediction, 2))
