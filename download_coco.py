import os
import requests
import zipfile
import tqdm
import shutil

def download_file(url, destination_path):
    """Downloads a file from a URL with a progress bar."""
    print(f"Downloading {url.split('/')[-1]} to {destination_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an exception for HTTP errors

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    with open(destination_path, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, desc=url.split('/')[-1]) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Finished downloading {url.split('/')[-1]}")

def extract_zip(zip_path, extract_dir):
    """Extracts a zip file to a specified directory."""
    print(f"Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of members for tqdm progress bar
        members = zip_ref.namelist()
        for member in tqdm.tqdm(members, desc=f"Extracting {os.path.basename(zip_path)}"):
            zip_ref.extract(member, extract_dir)
    print(f"Finished extracting {zip_path}")

def download_coco2017(base_dir="coco-dataset"):
    """
    Downloads and extracts the COCO 2017 dataset.

    Args:
        base_dir (str): The base directory where the dataset will be saved.
    """
    image_base_url = "http://images.cocodataset.org/zips/"
    annotation_base_url = "http://images.cocodataset.org/annotations/"

    # Define files to download
    files_to_download = {
        "images": {
            "train2017.zip": image_base_url + "train2017.zip",
            "val2017.zip": image_base_url + "val2017.zip",
        },
        "annotations": {
            "annotations_trainval2017.zip": annotation_base_url + "annotations_trainval2017.zip",
        }
    }

    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    print(f"COCO 2017 dataset will be saved to: {os.path.abspath(base_dir)}")

    # Download and extract images
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for filename, url in files_to_download["images"].items():
        zip_path = os.path.join(images_dir, filename)
        if not os.path.exists(zip_path.replace(".zip", "")): # Check if extracted folder exists
            download_file(url, zip_path)
            extract_zip(zip_path, images_dir)
            os.remove(zip_path) # Clean up zip file
        else:
            print(f"Skipping {filename}: already extracted in {images_dir}")

    # Download and extract annotations
    annotations_dir = os.path.join(base_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    for filename, url in files_to_download["annotations"].items():
        zip_path = os.path.join(annotations_dir, filename)
        if not os.path.exists(os.path.join(annotations_dir, "instances_train2017.json")):
            download_file(url, zip_path)
            extract_zip(zip_path, base_dir) # Extract to base_dir to get 'annotations' folder
            os.remove(zip_path) # Clean up zip file
        else:
            print(f"Skipping {filename}: annotations already extracted in {annotations_dir}")

    print("\nCOCO 2017 dataset download and extraction complete!")
    print(f"Dataset is located at: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    # Ensure requests and tqdm are installed
    try:
        import requests
        import tqdm
    except ImportError:
        print("Please install 'requests' and 'tqdm' libraries: pip install requests tqdm")
        exit(1)

    download_coco2017()
