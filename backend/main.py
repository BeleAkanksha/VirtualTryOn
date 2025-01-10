import cv2
import os
from tracker import FaceAndBodyTracker
from utils import (
    get_body_dimensions, overlay_image_on_body, load_dataset_images,
    select_camera, get_product_id_from_image, choose_image,
    track_and_recommend, display_recommended_items, choose_recommended_item
)

def display_recommended_items_with_overlay(recommended_items, frame, pose_landmarks):
    """Display and overlay recommended items on the body."""
    # print("Recommended Items:")
    for i, (idx, row) in enumerate(recommended_items.iterrows(), start=1):
        # print(f"{i}. {row['id']} {row['productDisplayName']}")
        # Load the recommended image
        recommended_image_path = f'./output_images/{row["id"]}.png'
        
        if os.path.exists(recommended_image_path):
            recommended_image = cv2.imread(recommended_image_path, cv2.IMREAD_UNCHANGED)
            # Check for alpha channel and overlay the image
            if pose_landmarks:
                body_dimensions = get_body_dimensions(pose_landmarks, frame.shape)
                if body_dimensions:
                    overlay_image_on_body(frame, recommended_image, body_dimensions)

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
            else:
                cap.release()
                cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()