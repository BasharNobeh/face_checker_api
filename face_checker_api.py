from flask import Flask, request, jsonify
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Initialize the face analysis
app_face = FaceAnalysis()
app_face.prepare(ctx_id=0, det_size=(640, 640))

# Reference image path (hardcoded, or you can make it a parameter)
reference_image_path = r"bashar.jpg"
reference_image = cv2.imread(reference_image_path)

if reference_image is None:
    raise FileNotFoundError(
        f"Reference image not found: {reference_image_path}")

# Perform face recognition on the reference image
faces1 = app_face.get(reference_image)

# Ensure at least one face is detected in the reference image
if not faces1:
    raise ValueError("No faces detected in the reference image.")

# Normalize the embedding of the reference image
embedding1 = faces1[0].embedding / np.linalg.norm(faces1[0].embedding)


def decode_base64_image(base64_string):
    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_string)
    # Convert bytes to numpy array and decode it into an image
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Get JSON data from the request
    data = request.get_json()
    student_id = data.get("student_id")
    base64_image = data.get("student_img")

    if not student_id or not base64_image:
        return jsonify({"error": "Student ID and base64 image data are required."}), 400

    # Decode the base64-encoded image
    image = decode_base64_image(base64_image)
    if image is None:
        return jsonify({"error": "Invalid image data."}), 400

    # Perform face recognition on the provided image
    faces2 = app_face.get(image)

    if not faces2:
        return jsonify({"error": "No face detected in the provided image."}), 400

    # Normalize the embedding of the current image
    embedding2 = faces2[0].embedding / np.linalg.norm(faces2[0].embedding)

    # Compute cosine similarity
    similarity = np.dot(embedding1, embedding2)

    # Convert similarity to percentage (0 to 100%)
    similarity_percentage = (similarity + 1) * 50
    similarity_percentage = np.clip(
        similarity_percentage, 0, 100)  # Clamp between 0 and 100

    result = {

        "similarity_percentage": f"{similarity_percentage:.2f}%"
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
