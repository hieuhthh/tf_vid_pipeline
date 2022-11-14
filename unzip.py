import zipfile
import os
import shutil

project_path = '/storage/hieunmt/zaloai_liveness'
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

# filename = 'public_test.zip'
# zip_file = os.path.join(from_download, filename)
# with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#     zip_ref.extractall(des)

filename = 'public_test_2.zip'
zip_file = os.path.join(from_download, filename)
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(des)

# filename = 'seed_42_20_fold_10.zip'
# zip_file = os.path.join(from_download, filename)
# with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#     zip_ref.extractall(des)