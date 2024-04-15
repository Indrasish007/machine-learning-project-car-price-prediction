from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
car = pd.read_csv("refine_car.csv")
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))


@app.route("/")
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()
    kms_driven = car["kms_driven"].unique()

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        year=year,
        fuel_types=fuel_type,
        kms_driven=kms_driven,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        companies = request.form.get("companies")
        car_models = request.form.get("car_models")
        year = int(request.form.get("year") or 0)
        fuel_type = request.form.get("fuel_type")
        kms_driven = int(request.form.get("kms_driven") or 0)
        print(companies, car_models, year, fuel_type, kms_driven)
        prediction = model.predict(
            pd.DataFrame(
                [[car_models, companies, year, kms_driven, fuel_type]],
                columns=["name", "company", "year", "kms_driven", "fuel_type"],
                dtype="object",
            )
        )

        return "{:,.2f}".format(prediction[0])
    except Exception as e:
        print(e)
        return {"error": "Something went wrong", "message": str(e)}


if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask
# import pandas as pd
# app = Flask(__name__)
# car=pd.read_csv('refine_car.csv')

# @app.route('/')
# def hello_world():
#     companies=sorted(car['company'].unique())
#     car_models=sorted(car['name'].unique())
#     year= sorted(car['year'].unique(),reverse=True)
#     fuel_type=car['fuel_type'].unique()
#     return str(len(car_models))

# if __name__=="__main__":
#      app.run(debug=True)
