import zipfile
import os
import shutil

project_path = '/home/lap14880/hieunmt/tf_vid_pipeline'
from_download = os.path.join(project_path, "download")

try:
    os.mkdir(project_path)
except:
    pass

des = os.path.join(project_path, "unzip")

try:
    os.mkdir(des)
except:
    pass

filename = 'seed_1024_20_fold_0.zip'
zip_file = os.path.join(from_download, filename)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(des)

filename = 'public_test.zip'
zip_file = os.path.join(from_download, filename)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(des)