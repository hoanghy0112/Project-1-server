from flask import request, Flask

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def login():
    return "Hello world"


@app.route("/login", methods=["POST"])
def login():
    return "Post an image"
