"""Flask web app for uploading an image and predicting emotion."""

from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

from predict import predict_image
from utils import save_uploaded_file


app = Flask(__name__)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image file."
        else:
            try:
                image_path = save_uploaded_file(file)
                result = predict_image(image_path)
                image_url = f"/uploads/{Path(image_path).name}"
            except Exception as exc:  # Keep message friendly for beginners
                error = f"Prediction failed: {exc}"

    return render_template("index.html", result=result, image_url=image_url, error=error)


if __name__ == "__main__":
    app.run(debug=True)
