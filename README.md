# Crimes-Against-Women
Here is a **README.md** file for your **GitHub repository**, summarizing your entire project, including **crime trend analysis, machine learning models, NLP algorithms, and predictive modeling**.  

---

## **README.md** - Exploratory Data Analysis, Trend Visualization, and Predictive Modeling of Crimes Against Women in India  

### 📌 **Project Overview**  
This project focuses on **Exploratory Data Analysis (EDA), trend visualization, and predictive modeling** of **Crimes Against Women in India (2001-2021)**. Using **machine learning, time-series forecasting (Prophet, LSTM), decision trees, and NLP algorithms**, we analyze crime trends, identify key patterns, and predict future occurrences.  

🚀 **Key Features:**  
✅ **Crime Trend Analysis** – Understanding past crime patterns with data visualization.  
✅ **Predictive Modeling** – Forecasting future crime rates using **Prophet & LSTM**.  
✅ **Machine Learning** – Decision Trees & Regression for crime classification.  
✅ **NLP Integration** – Extracting insights from text reports (NER, Sentiment Analysis, Crime Classification).  
✅ **State-Wise Analysis** – Identifying high-risk areas & crime-prone regions.  

---

## 🔍 **Datasets Used**  
The dataset contains **crime records from 2001 to 2021** collected from **KAGGLE**.  

📊 **Key Features in the Dataset:**  
- **State** – Location of the crime.  
- **Year** – Crime occurrence timeline.  
- **Crime Type** – Rape, Domestic Violence, Dowry Deaths, etc.  
- **Total Crimes** – Total number of cases reported.  

---

## 📊 **Exploratory Data Analysis (EDA)**  
✔ **Visualizing Crime Trends Over Time** (Line Graphs, Heatmaps).  
✔ **State-Wise Distribution of Crimes** (Bar Charts, Geospatial Maps).  
✔ **Seasonal Crime Variations** (Yearly & Monthly Trends).  

📌 **Key Findings:**  
- **Domestic Violence & Assault cases are the most reported crimes.**  
- **Crime rates have increased in metro cities like Delhi, Maharashtra, and UP.**  
- **Certain crimes exhibit seasonal trends (e.g., domestic violence increases during festivals).**  

---

## 🤖 **Machine Learning Models for Crime Prediction**  

### 📌 **1. Decision Tree for Crime Classification**  
- Classifies crime cases based on **state, year, and type**.  
- Helps identify **high-risk crime-prone regions**.  
- **Accuracy Achieved:** ~80%  

🔹 **Code Implementation:**  
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 📌 **2. Prophet Model for Time-Series Crime Forecasting**  
- Predicts **future crime trends** based on past data.  
- Automatically captures **seasonal variations & trends**.  
- **Best for long-term crime forecasting.**  

🔹 **Code Implementation:**  
```python
from prophet import Prophet

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=5, freq='Y')
forecast = model.predict(future)
```

---

### 📌 **3. LSTM (Long Short-Term Memory) for Deep Learning-Based Prediction**  
- **Neural Network-based model for crime time-series forecasting.**  
- Handles **long-term dependencies** in crime patterns.  
- **Best for short-term & fluctuating crime trends.**  

🔹 **Code Implementation:**  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

---


### 🎯 **Final Thoughts**  
This project **combines Machine Learning, Time-Series Forecasting, and NLP** to provide a **comprehensive crime analysis & prediction system**. The insights derived can **help law enforcement, policymakers, and researchers** take preventive measures against crimes against women. 🚀  
