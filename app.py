from flask import Flask,request, render_template
import numpy as np
import pandas as pd
import pickle

#loading models
dtr = pickle.load(open('models\\dtr.pkl','rb'))
preprocessor = pickle.load(open('models\\preprocessor.pkl','rb'))



# load CSV data and extract unique values for Area and Item
data = pd.read_csv('Dataset\\Final_Dataset_df.csv')  # Replace with your CSV path
areas = sorted(data['Area'].unique())
items = sorted(data['Item'].unique())


# flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Pass the unique Area and Item options to the frontend
    return render_template('index.html', areas=areas, items=items)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Create a feature array
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

        # Apply preprocessor transformation
        transformed_features = preprocessor.transform(features)

        # Make prediction
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        # Round prediction to nearest integer
        rounded_prediction = round(prediction[0][0])

        # Return the result and options again (to keep dropdowns populated)
        return render_template('index.html', prediction=rounded_prediction, areas=areas, items=items)

if __name__ == "__main__":
    app.run(debug=True)
