import os
import glob
from PIL import Image
from rembg import remove, new_session
import numpy as np
from tqdm import tqdm

def remove_background_all():
    """
    Remove background from all images in all 20 subfolders and organize them in a new folder structure
    Uses GPU acceleration for faster processing
    """
    
    # Input and output paths
    input_base_folder = "medicinal plant leaf set"
    output_base_folder = "removed background"
    
    # Create main output directory
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Initialize GPU session for faster processing
    print("Initializing GPU session for background removal...")
    try:
        # Try to use GPU with CUDA
        session = new_session("u2net", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("✓ GPU acceleration enabled (CUDA)")
    except Exception as e:
        # Fallback to CPU if GPU is not available
        session = new_session("u2net", providers=['CPUExecutionProvider'])
        print(f"⚠ GPU not available, using CPU: {e}")
    
    # Get all subfolders (plant categories)
    plant_categories = [f for f in os.listdir(input_base_folder) 
                       if os.path.isdir(os.path.join(input_base_folder, f))]
    
    print(f"Found {len(plant_categories)} plant categories: {plant_categories}")
    
    total_images = 0
    total_processed = 0
    
    # Process each plant category
    for category in plant_categories:
        print(f"\n{'='*50}")
        print(f"Processing category: {category}")
        print(f"{'='*50}")
        
        # Create output folder for this category
        category_output = os.path.join(output_base_folder, category)
        os.makedirs(category_output, exist_ok=True)
        
        # Get all images in this category
        category_path = os.path.join(input_base_folder, category)
        image_files = glob.glob(os.path.join(category_path, "*.jpg"))
        
        if not image_files:
            print(f"No images found in {category}")
            continue
        
        print(f"Found {len(image_files)} images in {category}")
        total_images += len(image_files)
        
        # Process each image with progress bar
        processed_count = 0
        for image_path in tqdm(image_files, desc=f"Processing {category}"):
            try:
                # Get filename
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Load image
                input_image = Image.open(image_path)
                
                # Remove background using GPU-accelerated session
                output_image = remove(input_image, session=session)
                
                # Save processed image (keep original filename but change extension to PNG)
                processed_save_path = os.path.join(category_output, f"{name_without_ext}.png")
                output_image.save(processed_save_path)
                
                processed_count += 1
                total_processed += 1
                
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                continue
        
        print(f"Successfully processed {processed_count}/{len(image_files)} images in {category}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total images found: {total_images}")
    print(f"Total images processed: {total_processed}")
    print(f"Success rate: {(total_processed/total_images)*100:.1f}%" if total_images > 0 else "No images found")
    print(f"Output folder: {output_base_folder}")
    print(f"\nAll background-removed images are now organized in the '{output_base_folder}' folder")
    print(f"Each plant category has its own subfolder with the processed images.")

if __name__ == "__main__":
    remove_background_all()
