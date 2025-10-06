import os
import requests
import tarfile
from tqdm import tqdm

# --- Configuration ---
# Set the path where you want to download and extract the files.
# The script will create this directory if it doesn't exist.
DESTINATION_PATH = "data/imagenet-c" 

# Set to True if you want to delete the .tar archives after extraction to save space.
REMOVE_TAR_AFTER_EXTRACTION = True
# ---------------------


def download_file(url, destination):
    """
    Downloads a file from a URL to a destination with a progress bar.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True,
                desc=f"Downloading {os.path.basename(destination)}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(destination):
            os.remove(destination) # Clean up partial download
        return False
    return True

def extract_tar(tar_path, extract_path):
    """
    Extracts a .tar file to a specified path.
    """
    print(f"Extracting {os.path.basename(tar_path)}...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
        return True
    except tarfile.TarError as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def main():
    """
    Main function to download and extract ImageNet-C corruptions.
    """
    # The 19 corruptions are grouped into 5 .tar archives.
    # This dictionary maps the archive file to the corruption folders it contains.
    CORRUPTION_GROUPS = {
        "blur.tar": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
        "digital.tar": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
        "extra.tar": ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
        "noise.tar": ["gaussian_noise", "shot_noise", "impulse_noise"],
        "weather.tar": ["snow", "frost", "fog", "brightness"]
    }
    
    base_url = "https://zenodo.org/record/2235448/files/"
    
    # Create the destination directory if it doesn't exist
    os.makedirs(DESTINATION_PATH, exist_ok=True)
    print(f"Files will be downloaded and extracted to: {os.path.abspath(DESTINATION_PATH)}\n")

    for archive_file, corruption_list in CORRUPTION_GROUPS.items():
        print("-" * 50)
        print(f"Processing archive: {archive_file}")

        # Check if all sub-folders for this archive already exist to skip the process
        all_extracted = all(
            os.path.exists(os.path.join(DESTINATION_PATH, "imagenet-c", corruption))
            for corruption in corruption_list
        )

        if all_extracted:
            print(f"All corruptions from {archive_file} seem to be extracted. Skipping.")
            continue

        # 1. Download the archive file
        tar_path = os.path.join(DESTINATION_PATH, archive_file)
        if not os.path.exists(tar_path):
            url = f"{base_url}{archive_file}?download=1"
            if not download_file(url, tar_path):
                continue # Skip to next archive if download failed
        else:
            print(f"{archive_file} already downloaded.")

        # 2. Extract the .tar file
        if not extract_tar(tar_path, DESTINATION_PATH):
            continue # Skip to next if extraction failed

        # 3. Optionally remove the .tar file to save space
        if REMOVE_TAR_AFTER_EXTRACTION:
            try:
                os.remove(tar_path)
                print(f"Removed archive: {tar_path}")
            except OSError as e:
                print(f"Error removing {tar_path}: {e}")

    print("\n" + "=" * 50)
    print("All available ImageNet-C corruption archives have been processed.")
    print("=" * 50)


if __name__ == "__main__":
    main()