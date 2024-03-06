import cv2
import os

def blur_images_in_folder(folder_path, output_folder, blur_strength=15):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more image extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Read the image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Apply Gaussian blur
            blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

            # Save the blurred image to the output folder
            output_path = os.path.join(output_folder, f"{filename}")
            cv2.imwrite(output_path, blurred_image)

if __name__ == "__main__":
    # Set your input and output folder paths
    input_folder_path = r"/mnt/c/Users/kirva/Desktop/Project_Design/Project_UAV/uavf_2024/tests/imaging/imaging_data/tile_dataset/images"
    output_folder_path = r"/mnt/c/Users/kirva/Desktop/Project_Design/Project_UAV/uavf_2024/tests/imaging/imaging_data/tile_dataset/blur_101_images"

    # Specify the blur strength (adjust as needed)
    blur_strength = 101

    # Blur the images in the folder
    blur_images_in_folder(input_folder_path, output_folder_path, blur_strength)
