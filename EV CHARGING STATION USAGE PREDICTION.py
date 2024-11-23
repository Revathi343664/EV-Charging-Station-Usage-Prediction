#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'EVChargingStationUsage.csv'  # Ensure the file is in the same directory or adjust the path accordingly
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(data.info())
print("\nSample Data:")
display(data.head())

# Data Cleaning and Preprocessing

# Handle missing values
# Option 1: Drop rows with missing values (if you want to only keep complete cases)
data_cleaned = data.dropna()

# Option 2: Fill missing values with appropriate values (you can adjust based on column types)
data_filled = data.fillna({
    'Fee': 0,  # Assuming 0 for missing Fee
    'Energy (kWh)': data['Energy (kWh)'].mean(),  # Fill with mean for numerical columns
    # Add other columns here as needed
})

# Encoding categorical variables
categorical_columns = ['Port Type', 'Plug Type', 'City', 'State/Province', 'Country', 'Currency', 'Ended By']
data_encoded = pd.get_dummies(data_filled, columns=categorical_columns)

# Feature Engineering

# Convert date and time columns to datetime format
data_encoded['Start Date'] = pd.to_datetime(data_encoded['Start Date'], errors='coerce')
data_encoded['End Date'] = pd.to_datetime(data_encoded['End Date'], errors='coerce')
data_encoded['Transaction Date (Pacific Time)'] = pd.to_datetime(data_encoded['Transaction Date (Pacific Time)'], errors='coerce')

# Calculate total usage duration in minutes
data_encoded['Total Duration (mins)'] = pd.to_timedelta(data_encoded['Total Duration (hh:mm:ss)']).dt.total_seconds() / 60
data_encoded['Charging Time (mins)'] = pd.to_timedelta(data_encoded['Charging Time (hh:mm:ss)']).dt.total_seconds() / 60

# Extract time-based features
data_encoded['Start Hour'] = data_encoded['Start Date'].dt.hour
data_encoded['Day of Week'] = data_encoded['Start Date'].dt.dayofweek
data_encoded['Month'] = data_encoded['Start Date'].dt.month

# Feature Scaling (for numerical columns)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['Energy (kWh)', 'GHG Savings (kg)', 'Gasoline Savings (gallons)', 'Latitude', 'Longitude', 'Total Duration (mins)', 'Charging Time (mins)']
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Display the cleaned and processed data
print("\nCleaned and Preprocessed Data:")
display(data_encoded.head())


# In[2]:


# Load the dataset
file_path = 'EVChargingStationUsage.csv'  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Display unique values in the "Station Name" column
unique_station_names = data['Station Name'].unique()
num_unique_station_names = data['Station Name'].nunique()

# Print results
print("Unique Station Names:")
print(unique_station_names)
print("\nNumber of Unique Station Names:", num_unique_station_names)


# In[3]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'EVChargingStationUsage.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(data.info())
print("\nSample Data:")
display(data.head())

# Data Cleaning and Preprocessing

# Handle missing values by filling or dropping (adjust based on column types and needs)
data_cleaned = data.fillna({
    'Fee': 0,  # Assuming 0 for missing Fee
    'Energy (kWh)': data['Energy (kWh)'].mean()  # Fill with mean for numerical columns
    # You can add more columns and logic as needed
})

# Convert categorical variables to category type and encode them
categorical_columns = ['Port Type', 'Plug Type', 'City', 'State/Province', 'Country', 'Currency', 'Ended By']
for col in categorical_columns:
    if col in data_cleaned.columns:
        data_cleaned[col] = data_cleaned[col].astype('category')

# Convert date and time columns to datetime format
data_cleaned['Start Date'] = pd.to_datetime(data_cleaned['Start Date'], errors='coerce')
data_cleaned['End Date'] = pd.to_datetime(data_cleaned['End Date'], errors='coerce')
data_cleaned['Transaction Date (Pacific Time)'] = pd.to_datetime(data_cleaned['Transaction Date (Pacific Time)'], errors='coerce')

# Convert 'Total Duration (hh:mm:ss)' and 'Charging Time (hh:mm:ss)' to minutes
if 'Total Duration (hh:mm:ss)' in data_cleaned.columns:
    data_cleaned['Total Duration (mins)'] = pd.to_timedelta(data_cleaned['Total Duration (hh:mm:ss)']).dt.total_seconds() / 60
else:
    print("Column 'Total Duration (hh:mm:ss)' is missing")

if 'Charging Time (hh:mm:ss)' in data_cleaned.columns:
    data_cleaned['Charging Time (mins)'] = pd.to_timedelta(data_cleaned['Charging Time (hh:mm:ss)']).dt.total_seconds() / 60
else:
    print("Column 'Charging Time (hh:mm:ss)' is missing")

# Extract time-based features
data_cleaned['Start Hour'] = data_cleaned['Start Date'].dt.hour
data_cleaned['Day of Week'] = data_cleaned['Start Date'].dt.day_name()
data_cleaned['Month'] = data_cleaned['Start Date'].dt.month

# Display the cleaned data to check if 'Port Type' is available
print("\nCleaned Data:")
display(data_cleaned.head())


# EDA

# 1. Usage Patterns by Hour
plt.figure(figsize=(10, 5))
sns.countplot(data=data_cleaned, x='Start Hour', palette='viridis')
plt.title('Usage Patterns by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Charging Sessions')
plt.show()

# Explanation: This plot identifies peak charging hours, which is crucial for scheduling and predicting occupancy patterns.

# 2. Usage by Day of the Week
plt.figure(figsize=(10, 5))
sns.countplot(data=data_cleaned, x='Day of Week', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='Set2')
plt.title('Usage Patterns by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Charging Sessions')
plt.show()

# Explanation: This plot highlights which days are busier, helping stakeholders plan station maintenance and predict usage trends.

# 3. Energy Usage Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data_cleaned['Energy (kWh)'], kde=True, bins=30, color='blue')
plt.title('Distribution of Energy Usage (kWh)')
plt.xlabel('Energy (kWh)')
plt.ylabel('Frequency')
plt.show()

# Explanation: This plot shows how much energy is typically consumed in charging sessions, important for understanding demand and planning infrastructure.

# 4. Charging Time Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data_cleaned['Charging Time (mins)'], kde=True, bins=30, color='green')
plt.title('Distribution of Charging Time')
plt.xlabel('Charging Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# Explanation: This plot helps analyze session duration trends, which can predict station occupancy over time.

# 5. Energy Usage by Port Type
if 'Port Type' in data_cleaned.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=data_cleaned, x='Port Type', y='Energy (kWh)')
    plt.title('Energy Usage by Port Type')
    plt.xlabel('Port Type')
    plt.ylabel('Energy (kWh)')
    plt.show()

# Explanation: This highlights energy consumption differences across port types, which can guide infrastructure upgrades.

# 6. Charging Time by Plug Type
if 'Plug Type' in data_cleaned.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data_cleaned, x='Plug Type', y='Charging Time (mins)', palette='pastel')
    plt.title('Charging Time by Plug Type')
    plt.xlabel('Plug Type')
    plt.ylabel('Charging Time (minutes)')
    plt.show()

# Explanation: This plot informs stakeholders about plug-specific charging behavior to optimize station design.

# 7. Geographic Analysis
if 'Latitude' in data_cleaned.columns and 'Longitude' in data_cleaned.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_cleaned, x='Longitude', y='Latitude', hue='City', legend=False, palette='Spectral')
    plt.title('Geographic Distribution of Charging Stations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Explanation: Visualizing the geographic spread of charging stations helps stakeholders identify underserved areas.

# 8. Total Duration by Day of Week
plt.figure(figsize=(10, 5))
sns.boxplot(data=data_cleaned, x='Day of Week', y='Total Duration (mins)', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Total Duration of Charging Sessions by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Total Duration (mins)')
plt.show()


# In[4]:


# Import necessary libraries
get_ipython().system('pip install xgboost')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'EVChargingStationUsage.csv'
data = pd.read_csv(file_path)

# Display dataset overview
print("Dataset Overview:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Data Cleaning and Preprocessing

# Handle missing values
data_cleaned = data.fillna({
    'Fee': 0,  # Assuming 0 for missing Fee
    'Energy (kWh)': data['Energy (kWh)'].mean()  # Fill with mean for numerical columns
})

# Convert categorical variables to category type
categorical_columns = ['Port Type', 'Plug Type', 'City', 'State/Province', 'Country', 'Currency', 'Ended By']
for col in categorical_columns:
    if col in data_cleaned.columns:
        data_cleaned[col] = data_cleaned[col].astype('category')

# Convert date and time columns to datetime format
data_cleaned['Start Date'] = pd.to_datetime(data_cleaned['Start Date'], errors='coerce')
data_cleaned['End Date'] = pd.to_datetime(data_cleaned['End Date'], errors='coerce')
data_cleaned['Transaction Date (Pacific Time)'] = pd.to_datetime(data_cleaned['Transaction Date (Pacific Time)'], errors='coerce')

# Create new features
if 'Total Duration (hh:mm:ss)' in data_cleaned.columns:
    data_cleaned['Total Duration (mins)'] = pd.to_timedelta(data_cleaned['Total Duration (hh:mm:ss)']).dt.total_seconds() / 60
if 'Charging Time (hh:mm:ss)' in data_cleaned.columns:
    data_cleaned['Charging Time (mins)'] = pd.to_timedelta(data_cleaned['Charging Time (hh:mm:ss)']).dt.total_seconds() / 60
if 'Start Date' in data_cleaned.columns:
    data_cleaned['Start Hour'] = data_cleaned['Start Date'].dt.hour
    data_cleaned['Day of Week'] = data_cleaned['Start Date'].dt.day_name()

# One-hot encode categorical variables, including Day of Week
data_encoded = pd.get_dummies(data_cleaned, columns=['Day of Week'] + categorical_columns, drop_first=True)

# Define features and target variable
features = ['Energy (kWh)', 'Charging Time (mins)', 'Start Hour', 'Latitude', 'Longitude']
# Include one-hot encoded Day of Week features
features += [col for col in data_encoded.columns if col.startswith('Day of Week_')]

# Ensure target column exists
target = 'Station Occupancy'  # Replace with actual column name for the target
if target not in data_encoded.columns:
    print(f"Target column '{target}' not found. Check your dataset.")
    exit()

# Split the data into training and testing sets
X = data_encoded[features]
y = data_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation

# Logistic Regression
print("\nTraining Logistic Regression Model...")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Random Forest Classifier
print("\nTraining Random Forest Model...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# XGBoost Classifier
print("\nTraining XGBoost Model...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Evaluate Models with Accuracy
print("\nModel Accuracy:")
print(f"Logistic Regression: {accuracy_score(y_test, y_pred_logreg):.2f}")
print(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"XGBoost: {accuracy_score(y_test, y_pred_xgb):.2f}")

# Plot ROC Curve
def plot_roc_curve(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    else:
        print(f"{model_name} does not support predict_proba for ROC curve.")

plt.figure(figsize=(10, 6))
plot_roc_curve(logreg, X_test, y_test, "Logistic Regression")
plot_roc_curve(rf, X_test, y_test, "Random Forest")
plot_roc_curve(xgb, X_test, y_test, "XGBoost")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
file_path = 'EVChargingStationUsage.csv'
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
# Fill missing values
data_cleaned = data.fillna({
    'Fee': 0,
    'Energy (kWh)': data['Energy (kWh)'].mean()
})

# Convert datetime columns
if 'Start Date' in data_cleaned.columns:
    data_cleaned['Start Date'] = pd.to_datetime(data_cleaned['Start Date'], errors='coerce')
if 'End Date' in data_cleaned.columns:
    data_cleaned['End Date'] = pd.to_datetime(data_cleaned['End Date'], errors='coerce')
if 'Transaction Date (Pacific Time)' in data_cleaned.columns:
    data_cleaned['Transaction Date (Pacific Time)'] = pd.to_datetime(data_cleaned['Transaction Date (Pacific Time)'], errors='coerce')

# Convert duration columns to minutes
if 'Total Duration (hh:mm:ss)' in data_cleaned.columns:
    data_cleaned['Total Duration (mins)'] = pd.to_timedelta(data_cleaned['Total Duration (hh:mm:ss)']).dt.total_seconds() / 60
if 'Charging Time (hh:mm:ss)' in data_cleaned.columns:
    data_cleaned['Charging Time (mins)'] = pd.to_timedelta(data_cleaned['Charging Time (hh:mm:ss)']).dt.total_seconds() / 60

# Extract time-based features
if 'Start Date' in data_cleaned.columns:
    data_cleaned['Start Hour'] = data_cleaned['Start Date'].dt.hour
    data_cleaned['Day of Week'] = data_cleaned['Start Date'].dt.day_name()
    data_cleaned['Month'] = data_cleaned['Start Date'].dt.month

# Create 'Station Occupancy' column
data_cleaned['Station Occupancy'] = (data_cleaned['Charging Time (mins)'] > 0).astype(int)

# Select features and target
features = [
    'Start Hour', 'Month', 'Energy (kWh)', 'GHG Savings (kg)', 
    'Gasoline Savings (gallons)', 'Latitude', 'Longitude'
]
target = 'Station Occupancy'

# Filter for existing columns
features = [feature for feature in features if feature in data_cleaned.columns]

# Check for necessary columns
if target not in data_cleaned.columns:
    raise KeyError(f"Target column '{target}' not found in the dataset.")

# Split the data into training and testing sets
X = data_cleaned[features]
y = data_cleaned[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Ensure feature importance values exist and are non-zero
if hasattr(model, "feature_importances_"):
    feature_importance = model.feature_importances_
    
    if feature_importance.sum() > 0:  # Check if there are non-zero importances
        feature_names = features

        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print("All feature importances are zero or not calculated.")
else:
    print("The model does not have a feature_importances_ attribute.")


# In[ ]:




