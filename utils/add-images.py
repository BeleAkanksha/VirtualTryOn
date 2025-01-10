import os
import pandas as pd

# Path to the folder where your images are stored
images_folder_path = "output_images"

# Load your dataset
styles_df = pd.read_csv("filtered_styles.csv")

# Function to check if image exists and return the path
def get_image_path(product_id):
    image_path = os.path.join(images_folder_path, f"{product_id}.png")
    # Check if the image file exists
    if os.path.exists(image_path):
        return image_path
    else:
        return None  # Return None if the image does not exist

# Apply the function to add 'image_path' column only for images that exist
styles_df['image_path'] = styles_df['id'].apply(get_image_path)

# Filter out rows where the 'image_path' is None (i.e., image does not exist)
styles_df = styles_df.dropna(subset=['image_path'])

# Save the updated CSV with image paths
styles_df.to_csv("updated_styles.csv", index=False)

# Print the number of rows with missing images
missing_images_count = styles_df.shape[0] - styles_df.dropna(subset=['image_path']).shape[0]
print(f"Number of rows with missing images: {missing_images_count}")
