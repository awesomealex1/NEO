import numpy as np
import os
from PIL import Image
from pathlib import Path

def convert_cifar10c_to_images(cifar_c_path, output_path):
    """
    Converts the CIFAR-10-C .npy files for all severities into a directory of .png images.

    Args:
        cifar_c_path (str): The path to the directory containing CIFAR-10-C .npy files.
        output_path (str): The path to the directory where images will be saved.
    """
    cifar_c_path = Path(cifar_c_path)
    output_path = Path(output_path)
    
    # Create the main output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_path}")

    # --- Load Labels ---
    labels_file = cifar_c_path / 'labels.npy'
    if not labels_file.exists():
        print(f"Error: labels.npy not found in {cifar_c_path}")
        return
    labels = np.load(labels_file)
    print("Labels loaded.")

    # --- Find and Process Corruption Files ---
    corruption_files = sorted([f for f in cifar_c_path.glob('*.npy') if f.name != 'labels.npy'])

    for npy_file in corruption_files:
        corruption_name = npy_file.stem
        print(f"\nProcessing corruption: {corruption_name}...")
        
        images = np.load(npy_file)
        
        # --- START: MODIFIED SECTION ---
        # Loop through all 5 severity levels
        for severity in range(1, 6):
            print(f"  - Processing Severity {severity}...")
            
            # Calculate the start and end indices for the current severity
            start_index = (severity - 1) * 10000
            end_index = severity * 10000
            
            # Process the 10,000 images for this severity
            for i in range(start_index, end_index):
                image_array = images[i]
                
                # The labels array corresponds to the original 10,000 test images.
                # Use modulo to get the correct label index for any severity.
                label_index = i % 10000
                label = labels[label_index]
                
                # Create directory structure: output / corruption / severity / label
                save_dir = output_path / corruption_name / str(severity) / str(label)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the image
                img = Image.fromarray(image_array)
                img.save(save_dir / f'image_{i}.png')
        # --- END: MODIFIED SECTION ---
            
    print("\nConversion complete!")

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Set the path to the directory where your CIFAR-10-C files are located.
    # This is the directory that contains files like 'gaussian_noise.npy', 'labels.npy', etc.
    cifar_c_source_dir = 'data/cifar10_c/CIFAR-10-C/'  # Change this to your CIFAR-10-C directory path
    
    # Set the path where you want to save the output images.
    output_images_dir = 'data/cifar-10-c/'
    
    convert_cifar10c_to_images(cifar_c_source_dir, output_images_dir)