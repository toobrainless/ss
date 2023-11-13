import os
import zipfile

import gdown

url = (
    "https://drive.google.com/file/d/1jaZZPGBAl61OHr_xJgb-m8jqn0qOmNdV/view?usp=sharing"
)
output = "spexp.zip"

gdown.download(url, output, fuzzy=True)

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall()

os.remove(output)
os.rename("spexp", "ss_model")
