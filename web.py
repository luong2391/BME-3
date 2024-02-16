from flask import Flask, render_template, request
import requests
from pharma import load_data, feature_engineering, split_data, train_model, evaluate_model, tune_model, print_score

app = Flask(__name__)

# Define the route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
# Modify the run_model function to capture the returned scores and pass them to the template
def run_model():
    if request.method == "POST":
        # Load data
        file_path = r"D:\Nam_3_2\BME\BME3\data_4_paper.xlsx"
        df = load_data(file_path)

        # Feature engineering
        df = feature_engineering(df)

        # Split data
        X_train, X_test, y_train, y_test = split_data(df, 'amount')

        # Train model
        model = train_model(X_train, y_train)

        # Tune model
        # tuned_model = tune_model(X_train, y_train)

        # Evaluate model and capture the scores
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_scores = print_score(y_train, train_pred)
        test_scores = print_score(y_test, test_pred)

        # Combine train and test scores
        result = {
            'Train Scores': train_scores,
            'Test Scores': test_scores
        }

        return render_template('result.html', result=result)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
