import gdown
import zipfile
import os

def download_dataset(drive_url, target_folder):
    print(f"Processing: {drive_url}")
    
    # Create the folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    if "drive/folders" in drive_url or "folder" in drive_url:
        print("Downloading individual files directly...")
        gdown.download_folder(drive_url, output=target_folder, quiet=False, use_cookies=False)
        print(f"\n[SUCCESS] Folder contents downloaded to: {target_folder}")
    else:
        print("\n[INFO] Detected a Drive FILE link.")
        zip_path = os.path.join(target_folder, "temp_dataset.zip")
        output = gdown.download(drive_url, zip_path, quiet=False, fuzzy=True)
        
        if not output:
            print("[ERROR] Download failed.")
            return

        print(f"\nUnzipping {output}...")
        try:
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(target_folder)
            print(f"[SUCCESS] Extracted to: {target_folder}")
            
            # Clean up the zip file
            os.remove(output)
            
        except zipfile.BadZipFile:
            print("[ERROR] The downloaded file was not a valid zip file.")
            print("Check if the file on Drive is actually a .zip archive.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    url = 'https://drive.google.com/drive/folders/1ARBY9cIGj_jigi5Y88CtUy-GMj2clrXj'
    
    # Where you want the data to go
    output_dir = './raw' 
    # ---------------------

    download_dataset(url, output_dir)