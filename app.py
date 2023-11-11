import streamlit as st
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming df is your training dataframe and df_test is your test dataframe
def train_model():
    # Read the datasets
    df = pd.read_csv('Training.csv')
    df_test = pd.read_csv('Testing.csv')

    # Data preprocessing here (as per your code above)
    df = df.drop('Unnamed: 133', axis=1)
    X_train = df.drop('prognosis', axis=1)
    y_train = df['prognosis']
    X_test = df_test.drop('prognosis', axis=1)
    y_test = df_test['prognosis']

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    model.fit(X_train, y_train)

    # Optionally calculate the accuracy on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Model trained with accuracy: {accuracy:.2f}")

    return model

# Train model on startup
model = train_model()

# Streamlit app
st.title("Medical Diagnosis App")

# Collect symptoms from the user
symptoms = {}
for i in range(1, 134):  # Assuming you have 133 symptoms
    symptoms[f'Symptom {i}'] = st.checkbox(f'Symptom {i}')

if st.button("Predict"):
    # Prepare symptoms for the model
    input_symptoms = list(map(int, list(symptoms.values())))
    
    # Make a prediction
    prediction = model.predict([input_symptoms])[0]

    # Display the result
    st.write(f"The predicted prognosis is: {prediction}")
