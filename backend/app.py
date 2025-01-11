# filepath: /backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import cv2
import numpy as np
from tracker import FaceAndBodyTracker
from recommendation import ItemRecommender
from utils import (
    get_body_dimensions, overlay_image_on_body, load_dataset_images,
    get_product_id_from_image, track_and_recommend
)

app = Flask(__name__)
CORS(app)

# Load dataset images and filenames
dataset_images, filenames = load_dataset_images('./static/output_images')

# Load the product data
products_df = pd.read_csv("updated_styles.csv")

@app.route('/api/products', methods=['GET'])
def get_products():
    products_dict = products_df.to_dict(orient='records')
    products_serializable = [
        {key: convert_types(value) for key, value in product.items()}
        for product in products_dict
    ]
    products = []
    products.extend(products_serializable)
    output_images_dir = "./static/output_images"
    filenames = os.listdir(output_images_dir)
    for filename in filenames:
        product_id = get_product_id_from_image(filename)
        if product_id is not None:
            products.append({
                'id': product_id,
                'image': f'/output_images/{filename}'.replace("\\", "/"),
                'name': filename
            })
    return jsonify(products)

@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = products_df['gender'].unique().tolist()
    return jsonify(categories)

@app.route('/api/products/<category>', methods=['GET'])
def get_products_by_category(category):
    filtered_products = products_df[(products_df['gender'] == category) & (products_df['usage'] == 'Casual')]
    products_dict = filtered_products.to_dict(orient='records')
    products_serializable = [
        {key: convert_types(value) for key, value in product.items()}
        for product in products_dict
    ]
    return jsonify(products_serializable)

@app.route('/api/recommendations/<int:product_id>', methods=['GET'])
def get_recommendations(product_id):
    recommended_items = track_and_recommend(product_id)
    recommendations = recommended_items.to_dict(orient='records')
    return jsonify(recommendations)

@app.route('/api/tryon', methods=['POST'])
def try_on():
    data = request.json
    product_id = data.get('product_id')
    camera_index = data.get('camera_index', 0)

    # Find the selected image
    selected_image_path = f'./static/output_images/{product_id}.png'
    if not os.path.exists(selected_image_path):
        return jsonify({'error': 'Image not found'}), 404

    selected_image = cv2.imread(selected_image_path, cv2.IMREAD_UNCHANGED)

    # Initialize the tracker
    tracker = FaceAndBodyTracker()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return jsonify({'error': 'Error opening camera'}), 500

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = tracker.find_face_and_pose(frame)

            if tracker.pose_results.pose_landmarks:
                body_dimensions = get_body_dimensions(tracker.pose_results.pose_landmarks, frame.shape)
                if body_dimensions:
                    overlay_image_on_body(frame, selected_image, body_dimensions)

            cv2.imshow('Virtual Try-On', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('Virtual Try-On', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return jsonify({'message': 'Try-on session ended'})

def convert_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def get_product_id_from_image(filename):
    return filename.split("_")[0] if "_" in filename else None

def track_and_recommend(current_item_id):
    recommender = ItemRecommender(styles_csv_path="updated_styles.csv", images_folder_path="output_images")
    recommended_items = recommender.recommend_similar_items(current_item_id)
    if recommended_items.empty:
        print("No recommended items found.")
        return pd.DataFrame()
    return recommended_items

if __name__ == '__main__':
    app.run(debug=True)