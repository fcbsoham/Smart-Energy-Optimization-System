from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px

app = Flask(__name__)

model = joblib.load("model.pkl")

df = pd.read_csv("smart_energy_subset.csv")

@app.route("/")
def home():

    hourly = df.groupby("hour")["meter_reading"].mean()

    fig = px.line(
        x=hourly.index,
        y=hourly.values,
        labels={"x":"Hour","y":"Energy"}
    )

    graph = fig.to_html(full_html=False)

    avg_energy = round(df["meter_reading"].mean(),2)

    return render_template(
        "index.html",
        avg_energy=avg_energy,
        graph=graph
    )

@app.route("/predict", methods=["GET","POST"])
def predict():

    prediction = None

    if request.method == "POST":

        data = [
            float(request.form["building_id"]),
            float(request.form["site_id"]),
            float(request.form["primary_use"]),
            float(request.form["square_feet"]),
            float(request.form["air_temperature"]),
            float(request.form["cloud_coverage"]),
            float(request.form["wind_speed"]),
            float(request.form["hour"]),
            float(request.form["weekday"])
        ]

        prediction = model.predict([data])[0]

    return render_template(
        "predict.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)