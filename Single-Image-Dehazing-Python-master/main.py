import cv2
import numpy as np
from image_dehazer import image_dehazer

if __name__ == "__main__":
    image_path = 'Images/Foggy_bench.jpg'  # Ensure this file exists
    output_dir = 'outputImages/'  # Ensure this directory exists

    # Read input image
    HazeImg = cv2.imread(image_path)
    if HazeImg is None:
        print(f"Error: Unable to load image at {image_path}")
        exit(1)

    cv2.imshow('Hazy Image', HazeImg)

    # Create an instance of the dehazer
    dehazer = image_dehazer(showHazeTransmissionMap=False)

    # Remove haze
    HazeCorrectedImg, haze_map = dehazer.remove_haze(HazeImg)

    # Display results
    cv2.imshow('Haze Transmission Map', (haze_map * 255).astype(np.uint8))
    cv2.imshow('Enhanced Image', HazeCorrectedImg)

    # Save results
    cv2.imwrite(f"{output_dir}result.png", HazeCorrectedImg)
    print(f"Dehazed image saved to {output_dir}result.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
