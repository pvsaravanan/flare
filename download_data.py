import requests
import zipfile
import io
import os

url = "https://github.com/fisher85/ml-cybersecurity/blob/master/python-web-attack-detection/datasets/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.zip?raw=true"
save_path = "c:/proj/flare/data/"
zip_name = "dataset.zip"

os.makedirs(save_path, exist_ok=True)

print(f"Downloading fro {url}...")
try:
    r = requests.get(url)
    r.raise_for_status()
    print("Download complete. Extracting...")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(save_path)
    print("Extraction complete.")
    print("Files:", os.listdir(save_path))
except Exception as e:
    print(f"Error: {e}")
