import os
import zipfile

import gdown

url = (
    "https://drive.google.com/file/d/1750C4PQNwMTWhcu2jXgN6Qm--ZdzquIB/view?usp=sharing"
)
output = "spexp.zip"

gdown.download(url, output, fuzzy=True)

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall()

os.remove(output)
os.rename("best_model", "ss_model")
