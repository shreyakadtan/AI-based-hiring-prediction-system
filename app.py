from flask import Flask, render_template, request
from model import predict_candidate

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    skills = request.form["skills"]
    experience = float(request.form["experience"])
    education = request.form["education"]
    certifications = request.form["certifications"]
    projects = int(request.form["projects"])
    salary = float(request.form["salary"])

    result, probability = predict_candidate(
        skills, experience, education,
        certifications, projects, salary
    )

    return render_template(
        "index.html",
        prediction=result,
        prob=probability
    )


if __name__ == "__main__":
    app.run(debug=True)