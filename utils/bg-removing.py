import os
import shutil
from rembg import remove
from PIL import Image
from io import BytesIO

# Function to process all images in a folder
def remove_background_from_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each file in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        # Save output as PNG, even if input is JPG
        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".png")

        # Process only image files
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                with open(input_path, "rb") as input_file:
                    input_image = input_file.read()
                    output_image = remove(input_image)

                # Convert to PIL Image and save as PNG to preserve transparency
                img = Image.open(BytesIO(output_image)).convert("RGBA")
                img.save(output_path, format="PNG")
                print(f"Processed: {file_name}")

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

# Main function
def main():
    # Define input and output folder paths
    root_folder = os.getcwd()  # Current working directory
    input_folder = os.path.join(root_folder, "filtered")
    output_folder = os.path.join(root_folder, "output_images")

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist. Create the folder and add images.")
        return

    # Process images
    remove_background_from_images(input_folder, output_folder)
    print(f"Background removal completed. Processed images saved in '{output_folder}'.")

if __name__ == "__main__":
    main()
