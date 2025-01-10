# virtual-try-on

## Overview
The Virtual Try-On project is an application that allows users to virtually try on clothing items using their webcam. The application uses computer vision techniques to detect the user's body and overlay selected clothing items onto the user's image in real-time. Additionally, the application can recommend similar items based on the user's current selection.

## Features
- Real-time body and face detection using MediaPipe.
- Overlay clothing items onto the user's body with proper scaling and transparency.
- Recommend similar clothing items based on the current selection.
- Interactive user interface to select and switch between different clothing items.

## Project Structure
The project is organized into the following files and directories:

- `main.py`: The main script that runs the application.
- `tracker.py`: Contains the `FaceAndBodyTracker` class for detecting face and body landmarks.
- `utils.py`: Utility functions for image processing, camera selection, and item recommendation.
- `recommendation.py`: Contains the `ItemRecommender` class for recommending similar items.
- `output_images/`: Directory containing the clothing item images.
- `updated_styles.csv`: CSV file containing metadata for the clothing items.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/virtual-try-on.git
    cd virtual-try-on
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the main script:
    ```bash
    python main.py
    ```

2. Follow the on-screen instructions to select a camera and an initial clothing item.

3. Press 'r' to get recommendations for similar items.

4. Select a recommended item to try it on virtually.

## Dependencies
- OpenCV
- MediaPipe
- Pandas

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.