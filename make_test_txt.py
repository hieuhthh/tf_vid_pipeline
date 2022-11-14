import os
import numpy
import shutil

project_path = '/storage/hieunmt/zaloai_liveness/unzip'

def convert_test_txt(route, name):
    with open(name, "w") as f:
        for file in os.listdir(route):
            path = os.path.join(route, file)
            f.write(f"{path} {file}\n")

route = f'{project_path}/public/videos'
name = "public_test.txt"
convert_test_txt(route, name)

route = f'{project_path}/public_test_2/videos'
name = "public_test_2.txt"
convert_test_txt(route, name)