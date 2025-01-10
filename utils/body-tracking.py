import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from recommendation import ItemRecommender

class FaceAndBodyTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        self.mp_draw = mp.solutions.drawing_utils
        self.user_data = {
            'liked_items': [],
            'tryon_history': []
        }

    def find_face_and_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.face_results = self.face_detection.process(img_rgb)
        self.pose_results = self.pose.process(img_rgb)
        
        # Draw face detection bounding box
        if self.face_results.detections:
            for detection in self.face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
        
        # Draw pose landmarks
        if self.pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return img

    def track_user_interaction(self, item_id, frame, pose_landmarks):
        """Track the user interacting with a product."""
        self.user_data['tryon_history'].append(item_id)
        if item_id not in self.user_data['liked_items']:
            self.user_data['liked_items'].append(item_id)

        # Call recommend_items here right after the item is selected
        self.recommend_items(item_id)

    def recommend_items(self, current_item_id):
        """Recommend similar items based on the selected product."""
        recommender = ItemRecommender(styles_csv_path="updated_styles.csv", images_folder_path="output_images")
        
        # Call the recommend_similar_items method from ItemRecommender class
        recommended_items = recommender.recommend_similar_items(current_item_id)
        
        # Check if no recommendations are found
        if recommended_items.empty:
            print("No similar items found.")
            return []
        
        return recommended_items

def get_body_dimensions(pose_landmarks, frame_shape):
    if pose_landmarks:
        # Extract key landmarks
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        ih, iw, _ = frame_shape

        # Calculate shoulder width
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * iw

        # Calculate hip width with a larger expansion factor
        hips_width = abs(right_hip.x - left_hip.x) * iw
        expanded_hips_width = hips_width * 1.5  # Expand hips width by 50%

        # Ensure the T-shirt width is at least 50% of the frame width
        min_width = iw * 0.5
        body_width = int(max(shoulder_width, expanded_hips_width, min_width))

        # Calculate torso height
        torso_height = abs(left_hip.y - left_shoulder.y) * ih
        body_height = int(torso_height * 1.2)  # Add 20% height for better coverage

        # Center the T-shirt horizontally based on shoulder midpoint
        x_center = int((left_shoulder.x + right_shoulder.x) / 2 * iw)
        x_position = int(x_center - body_width // 2)

        # Adjust the T-shirt to start higher above the shoulders
        y_top = int(left_shoulder.y * ih) - int(0.2 * body_height)  # Move up by 20% of height
        y_position = max(0, y_top)  # Ensure it's not out of frame

        # Ensure valid positions
        x_position = max(0, min(x_position, iw - body_width))
        y_position = max(0, min(y_position, ih - body_height))

        return (x_position, y_position, body_width, body_height)
    return None

def overlay_image_on_body(frame, overlay_img, body_dimensions):
    x, y, w, h = body_dimensions
    ih, iw, _ = frame.shape

    # Ensure dimensions are valid and fit within the frame
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = min(w, iw - x)
    h = min(h, ih - y)

    if w <= 0 or h <= 0:
        print(f"Invalid overlay dimensions: width={w}, height={h}")
        return

    # Resize overlay to fit the target area
    overlay_resized = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)

    if overlay_resized.shape[2] == 4:  # Has alpha channel
        bgr_overlay = overlay_resized[:, :, :3]
        alpha_channel = overlay_resized[:, :, 3]

        # Resize alpha channel if needed
        alpha_channel_resized = cv2.resize(alpha_channel, (w, h), interpolation=cv2.INTER_AREA)

        # Normalize alpha and compute inverse
        alpha_mask = alpha_channel_resized / 255.0
        inverse_alpha_mask = 1.0 - alpha_mask

        # Get region of interest (ROI) in the frame
        roi = frame[y:y+h, x:x+w]
        if roi.shape[:2] != (h, w):
            print(f"ROI shape mismatch: Expected ({h}, {w}), Got {roi.shape[:2]}")
            return

        # Blend each color channel
        for c in range(3):  # Iterate over B, G, R
            roi[:, :, c] = (alpha_mask * bgr_overlay[:, :, c] +
                            inverse_alpha_mask * roi[:, :, c])

        # Place blended result back in frame
        frame[y:y+h, x:x+w] = roi
    else:
        frame[y:y+h, x:x+w] = overlay_resized

def load_dataset_images(directory):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.lower().endswith((".png")):
            # Load with alpha channel if available
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Ensure image is BGRA; add alpha if missing
                if img.shape[2] == 3:  # If no alpha channel
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                images.append(img)
                filenames.append(filename)
    return images, filenames

def choose_recommended_item(recommended_items):
    while True:
        try:
            selected_number = int(input(f"Select an item by number (1 to {len(recommended_items)}): ")) - 1
            if 0 <= selected_number < len(recommended_items):
                return recommended_items.iloc[selected_number]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def track_and_recommend(current_item_id):
    """Recommend similar items and return them."""
    recommender = ItemRecommender(styles_csv_path="updated_styles.csv", images_folder_path="output_images")
    
    recommended_items = recommender.recommend_similar_items(current_item_id)
    
    if recommended_items.empty:  # Check if there are no recommendations
        print("No recommended items found.")
        return []
    
    return recommended_items

def display_recommended_items(recommended_items):
    # Display numbered recommended items with number, index, and product display name
    print("Recommended Items:")
    for i, (idx, row) in enumerate(recommended_items.iterrows(), start=1):  # Start numbering from 1
        print(f"{i}. {row['id']} {row['productDisplayName']}")

def display_recommended_items_with_overlay(recommended_items, frame, pose_landmarks):
    """Display and overlay recommended items on the body."""
    print("Recommended Items:")
    for i, (idx, row) in enumerate(recommended_items.iterrows(), start=1):
        print(f"{i}. {row['id']} {row['productDisplayName']}")
        # Load the recommended image
        recommended_image_path = f'./output_images/{row["id"]}.png'
        
        if os.path.exists(recommended_image_path):
            recommended_image = cv2.imread(recommended_image_path, cv2.IMREAD_UNCHANGED)
            # Check for alpha channel and overlay the image
            if pose_landmarks:
                body_dimensions = get_body_dimensions(pose_landmarks, frame.shape)
                if body_dimensions:
                    overlay_image_on_body(frame, recommended_image, body_dimensions)

def list_available_cameras():
    index = 0
    arr = []
    max_tested = 10  # Limit the number of cameras to check
    while index < max_tested:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            cap.release()
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def select_camera():
    cameras = list_available_cameras()
    if len(cameras) == 0:
        print("No camera detected!")
        return None
    
    print("Available cameras:")
    for idx, camera in enumerate(cameras):
        print(f"{idx}: Camera {camera}")

    while True:
        try:
            selected_index = int(input(f"Select the camera index (0 to {len(cameras)-1}): "))
            if 0 <= selected_index < len(cameras):
                return cameras[selected_index]
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

def get_product_id_from_image(image_name, styles_csv_path="updated_styles.csv"):
    # Load the styles dataset
    df = pd.read_csv(styles_csv_path)
    
    # Extract the id corresponding to the selected image name (e.g., 123.jpg)
    product_id = image_name.split('.')[0]  # Assuming image name is in format "123.jpg"
    
    # Check if the id exists in the dataset
    product_row = df[df['id'] == int(product_id)]
    if not product_row.empty:
        return product_row.iloc[0]['id']
    else:
        return None

def choose_image(filenames):
    print("Available images:")
    for idx, name in enumerate(filenames):
        print(f"{idx}: {name}")
    
    while True:
        try:
            selected_index = int(input(f"Select an image by index (0 to {len(filenames) - 1}): "))
            if 0 <= selected_index < len(filenames):
                return selected_index
            else:
                print("Invalid index. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

def main():
    camera_index = select_camera()
    if camera_index is None:
        return

    dataset_images, filenames = load_dataset_images('./output_images')
    if not dataset_images:
        print("No images found in the dataset.")
        return

    selected_index = choose_image(filenames)
    selected_image = dataset_images[selected_index]
    selected_image_name = filenames[selected_index]

    # Get the product ID from the image filename
    current_item_id = get_product_id_from_image(filenames[selected_index])
    if current_item_id is None:
        print("Could not find product ID for the selected image.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    # Initialize the tracker
    tracker = FaceAndBodyTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
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
            print("Exiting...")
            break
        elif key == ord('r'):  # Trigger recommendation on pressing 'r'
            recommended_items = track_and_recommend(current_item_id)

            # Display the recommended items
            display_recommended_items(recommended_items)

            # Display and overlay the recommended items
            display_recommended_items_with_overlay(recommended_items, frame, tracker.pose_results.pose_landmarks)

            # Ask the user if they want to try on a recommended item
            user_input = input("Would you like to try on one of the recommended items? (y/n): ").strip().lower()
            if user_input == 'y':
                selected_row = choose_recommended_item(recommended_items)
                selected_recommended_item_id = selected_row['id']
                selected_image_path = f'./output_images/{selected_recommended_item_id}.png'

                # Check if the image file exists before trying to load it
                if os.path.exists(selected_image_path):
                    selected_image = cv2.imread(selected_image_path, cv2.IMREAD_UNCHANGED)  # Read the new image
                    print(f"Switched to recommended item {selected_recommended_item_id} for try-on.")
                else:
                    print(f"Image for item {selected_recommended_item_id} does not exist.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()