# embedding_server.py
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("clip-ViT-B-32")

@app.route("/embed", methods=["POST"])
def embed():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(BytesIO(image_file.read()))
    embedding = model.encode(image).tolist()
    return jsonify({"embedding": embedding})

