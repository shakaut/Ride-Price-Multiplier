Test the App Here: https://ride-price-multiplier.streamlit.app/

# Ride Price Multiplier Prediction

This project predicts the **price multiplier** for a ride-hailing platform based on demand–supply conditions such as active drivers, pending requests, distance, traffic, and weather. It uses a **custom Random Forest implementation** and provides an interactive **Streamlit web app** for predictions.  

---

## 📌 Problem Statement  
Ride-hailing platforms like Uber, Pathao often adjust ride prices dynamically. When demand exceeds supply (e.g., more requests than available drivers), prices increase.  
This project builds a **machine learning model** to predict the **price multiplier** given:  

- Area ID  
- Distance of ride (km)  
- Active drivers  
- Pending requests  
- Traffic conditions  
- Weather  
- Hour of day  
- Day of week  
- Event status (holiday/concert/sports event, etc.)  

---

## Features  

- **Custom Random Forest Regressor** (MyRandomForestRegressor) built from scratch.  
- Encodes categorical features using **preprocessing pipelines**.  
- Trained on **10,000+ rows of synthetic ride-hailing dataset**.  
- Model achieves good **RMSE performance**.  
- **Streamlit app** for user-friendly predictions.  

---

## Tech Stack  

- **Python**  
- **scikit-learn** (Decision Trees, preprocessing)  
- **NumPy & Pandas** (data handling)  
- **Joblib** (model persistence)  
- **Streamlit** (interactive app)  

---

## Project Structure  

- **Ride_Pricing_Prediction.ipynb**      Data generate, model building and prediction
- **app.py**                             Streamlit app
- **rf_model.pkl**                       Trained model 
- **preprocessor.pkl**                   Saved encoder/preprocessor
- **requirements.txt**                   Dependencies
- **README.md**                          Project documentation
