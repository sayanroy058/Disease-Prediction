from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

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
    print(f"Model trained with accuracy: {accuracy:.2f}")

    return model

# Train model on startup
model = train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get symptoms from form and prepare them for the model
        symptoms = request.form.to_dict()
        symptoms = list(map(int, symptoms.values()))  # Example preprocessing
        prediction = model.predict([symptoms])
        return render_template('result.html', prediction=prediction[0])
    return render_template('index.html')
