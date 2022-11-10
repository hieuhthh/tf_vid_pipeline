import os
import numpy
import shutil

route = '/home/lap14880/hieunmt/tf_vid_pipeline/unzip/public/videos'

def convert_test_txt(route):
    with open(f"test.txt", "w") as f:
        for file in os.listdir(route):
            path = os.path.join(route, file)
            f.write(f"{path} {file}\n")

convert_test_txt(route)