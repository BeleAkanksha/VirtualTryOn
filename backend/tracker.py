import cv2
import mediapipe as mp
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