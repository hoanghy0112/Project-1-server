from flask import request, Flask

from run_model import run, model, postprocessor

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return "Home"


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    ext = image.filename.split(".")[-1]
    filename = f"download_temp.{ext}"
    image.save(filename)
    result = run(model, postprocessor, filename)
    return result

