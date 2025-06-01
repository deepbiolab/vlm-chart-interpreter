from huggingface_hub import snapshot_download
import os
import zipfile

def find_and_extract_zips(directory):
    """Recursively find and extract all zip files in the directory"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_dir = os.path.dirname(zip_path)  # Extract to same directory as zip
                print(f"Extracting {zip_path} to {extract_dir}")
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"✅ Successfully extracted {file}")

                except Exception as e:
                    print(f"❌ Error extracting {file}: {str(e)}")

def download_and_extract_dataset():
    print("Downloading dataset...")
    snapshot_download(
        repo_id="listen2you002/ChartLlama-Dataset", 
        repo_type="dataset",
        local_dir="dataset",
        local_dir_use_symlinks=False 
    )
    
    print("\nSearching for and extracting zip files...")
    find_and_extract_zips("dataset")

if __name__ == "__main__":
    download_and_extract_dataset()