#flask scikit-learn pandas pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np


app= Flask(__name__)

data = pd.read_csv("cleaned_data.csv")
pipe = pickle.load(open('RidgeModel', 'rb'))

@app.route('/')
def index():


    locations = sorted(data["location"].unique())
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    location = request.form.get("location") # request location from index
    bhk = request.form.get("bhk") # no of bedroom
    bath = request.form.get("bath")# no of bathrooms,hall,kitchen
    sqft = request.form.get("total_sqft")# how much sqft

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    prediction = pipe.predict(input)[0] *1e5

    return str(np.round(prediction, 2))
    #<br> use for breaking the line in html


if __name__== "__main__":
    app.run(debug=True, port=5001)

