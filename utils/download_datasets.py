import os
import requests
import tarfile
import subprocess
import platform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
DESTINATION_PATH = "data/imagenet-c"
REMOVE_TAR_AFTER_EXTRACTION = True
MAX_WORKERS = 5  # Run all 5 groups simultaneously
# ---------------------

CORRUPTION_GROUPS = {
    "blur.tar": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "digital.tar": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "extra.tar": ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
    "noise.tar": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "weather.tar": ["snow", "frost", "fog", "brightness"]
}

BASE_URL = "https://zenodo.org/record/2235448/files/"

def download_file(url, destination):
    """Downloads a file with a progress bar (position handling for threads)."""
    try:
        # Check if file exists and has size
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f"Skipping download (exists): {os.path.basename(destination)}")
            return True

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            # Use distinct positions for bars so they don't overlap in the terminal
            thread_id = int(os.getpid()) # Simplified; just for unique desc
            desc = f"DL {os.path.basename(destination)}"
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True,
                desc=desc, leave=False
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    return True

def extract_tar_fast(tar_path, extract_path):
    """
    Tries to use system 'tar' for speed (Linux/Mac), falls back to Python tarfile.
    Python's tarfile module is notoriously slow compared to native C implementation.
    """
    filename = os.path.basename(tar_path)
    print(f"Extracting {filename}...")
    
    # Attempt system tar (much faster on EPYC/Linux)
    if platform.system() != "Windows":
        try:
            subprocess.run(
                ["tar", "-xf", tar_path, "-C", extract_path], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            print(f"Extraction complete (System Tar): {filename}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"System tar failed for {filename}, falling back to Python native...")

    # Fallback to Python native
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)
        print(f"Extraction complete (Python): {filename}")
        return True
    except tarfile.TarError as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def process_archive(archive_name, corruption_list):
    """
    Worker function to handle the full lifecycle of one archive.
    """
    # 1. Check existing
    all_extracted = all(
        os.path.exists(os.path.join(DESTINATION_PATH, corruption))
        for corruption in corruption_list
    )
    if all_extracted:
        return f"{archive_name}: Already extracted."

    # 2. Download
    tar_path = os.path.join(DESTINATION_PATH, archive_name)
    url = f"{BASE_URL}{archive_name}?download=1"
    
    if not download_file(url, tar_path):
        return f"{archive_name}: Download failed."

    # 3. Extract
    if not extract_tar_fast(tar_path, DESTINATION_PATH):
        return f"{archive_name}: Extraction failed."

    # 4. Cleanup
    if REMOVE_TAR_AFTER_EXTRACTION:
        try:
            os.remove(tar_path)
        except OSError:
            pass
            
    return f"{archive_name}: Processed successfully."

def main():
    os.makedirs(DESTINATION_PATH, exist_ok=True)
    print(f"Starting parallel processing on {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_archive = {
            executor.submit(process_archive, archive, clist): archive 
            for archive, clist in CORRUPTION_GROUPS.items()
        }
        
        for future in as_completed(future_to_archive):
            archive = future_to_archive[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"{archive} generated an exception: {e}")

if __name__ == "__main__":
    main()