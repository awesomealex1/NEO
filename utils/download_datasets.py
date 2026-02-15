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
MAX_WORKERS = 5  
EXPECTED_FILES_PER_CORRUPTION = 250000 # 50k images * 5 severities
# ---------------------

CORRUPTION_GROUPS = {
    "blur.tar": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "digital.tar": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
    "extra.tar": ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
    "noise.tar": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "weather.tar": ["snow", "frost", "fog", "brightness"]
}

BASE_URL = "https://zenodo.org/record/2235448/files/"

def count_files_fast(dir_path):
    """
    recursively counts files in a directory using os.scandir for speed.
    """
    if not os.path.exists(dir_path):
        return 0
    
    count = 0
    try:
        # scan directory content
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file():
                    count += 1
                elif entry.is_dir():
                    count += count_files_fast(entry.path)
    except OSError:
        pass # Handle permission errors or disappearing files
    return count

def download_file(url, destination):
    """Downloads a file with a progress bar."""
    try:
        # Check if archive exists and has size
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f"Archive exists, skipping download: {os.path.basename(destination)}")
            return True

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
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
    """
    filename = os.path.basename(tar_path)
    print(f"Extracting {filename}...")
    
    if platform.system() != "Windows":
        try:
            # -x: extract, -f: file, -C: change directory before extracting
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
    # 1. Check file counts for ALL corruptions in this archive
    # If all corruptions inside this tar have > 250k files, we skip the whole thing.
    all_corruptions_ready = True
    
    for corruption in corruption_list:
        folder_path = os.path.join(DESTINATION_PATH, corruption)
        
        # Quick check: if folder doesn't exist, we definitely need to download
        if not os.path.exists(folder_path):
            all_corruptions_ready = False
            break
            
        # Deep check: count files
        print(f"Checking file count for {corruption}...")
        file_count = count_files_fast(folder_path)
        
        if file_count < EXPECTED_FILES_PER_CORRUPTION:
            print(f"Found only {file_count} files in {corruption}. Re-downloading archive.")
            all_corruptions_ready = False
            break
        else:
            print(f"{corruption} verified: {file_count} files.")

    if all_corruptions_ready:
        return f"SKIP {archive_name}: All contents verified."

    # 2. Download
    tar_path = os.path.join(DESTINATION_PATH, archive_name)
    url = f"{BASE_URL}{archive_name}?download=1"
    
    if not download_file(url, tar_path):
        return f"FAIL {archive_name}: Download failed."

    # 3. Extract
    if not extract_tar_fast(tar_path, DESTINATION_PATH):
        return f"FAIL {archive_name}: Extraction failed."

    # 4. Cleanup
    if REMOVE_TAR_AFTER_EXTRACTION:
        try:
            os.remove(tar_path)
        except OSError:
            pass
            
    return f"DONE {archive_name}: Processed successfully."

def main():
    os.makedirs(DESTINATION_PATH, exist_ok=True)
    print(f"Verifying {DESTINATION_PATH}...")
    print(f"Starting parallel processing on {MAX_WORKERS} workers...")
    print("-" * 60)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_archive = {
            executor.submit(process_archive, archive, clist): archive 
            for archive, clist in CORRUPTION_GROUPS.items()
        }
        
        for future in as_completed(future_to_archive):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main()