import urllib.request
import zipfile
import os

url = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
zip_path = "cell_images.zip"
extract_dir = "data"

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

if not os.path.exists(os.path.join(extract_dir, "cell_images")):
    if not os.path.exists(zip_path):
        print("Downloading cell_images.zip from NIH (this may take a few minutes)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    
    # Clean up zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
else:
    print("Dataset already exists in data/cell_images")
