import os
import glob
from PIL import Image
from rembg import remove
import numpy as np

def remove_background_test():
    """
    Remove background from first 100 images in Amla folder and save both versions for comparison
    """
    
    # Input and output paths
    input_folder = "medicinal plant leaf set/Amla"
    output_folder = "background_removal_test"
    
    # Create output directories
    original_output = os.path.join(output_folder, "with_background")
    processed_output = os.path.join(output_folder, "without_background")
    
    os.makedirs(original_output, exist_ok=True)
    os.makedirs(processed_output, exist_ok=True)
    
    # Get all jpg files from Amla folder
    image_files = glob.glob(os.path.join(input_folder, "*.jpg"))
    image_files.sort()  # Sort to ensure consistent order
    
    # Take only first 100 images
    test_images = image_files[:100]
    
    print(f"Found {len(image_files)} images in {input_folder}")
    print(f"Processing first {len(test_images)} images...")
    
    processed_count = 0
    
    for i, image_path in enumerate(test_images):
        try:
            # Get filename
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            print(f"Processing {i+1}/100: {filename}")
            
            # Load image
            input_image = Image.open(image_path)
            
            # Save original image to comparison folder
            original_save_path = os.path.join(original_output, filename)
            input_image.save(original_save_path)
            
            # Remove background
            output_image = remove(input_image)
            
            # Save processed image
            processed_save_path = os.path.join(processed_output, f"{name_without_ext}_nobg.png")
            output_image.save(processed_save_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count}/100 images")
    print(f"Original images saved to: {original_output}")
    print(f"Background-removed images saved to: {processed_output}")
    print(f"\nYou can now compare the images in both folders to see the results.")

if __name__ == "__main__":
    remove_background_test()
