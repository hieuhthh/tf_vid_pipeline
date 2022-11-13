import os
import numpy
import shutil

route = '/home/lap14880/hieunmt/tf_vid_pipeline/unzip/seed_128_10_fold_0'

def convert_txt(route, file_txt, stage):
    with open(file_txt, "r") as f:
        data = f.read()

    with open(f"{stage}.txt", "w") as f:
        data = data.split("\n")
        for line in data:
            try:
                name, label = line.split(' ')
                f_path = os.path.join(route, stage, name)
            except:
                pass

            f.write(f"{f_path} {label}\n")

file_txt = os.path.join(route, 'train_video.txt')
stage = "train"
convert_txt(route, file_txt, stage)

file_txt = os.path.join(route, 'val_video.txt')
stage = "val"
convert_txt(route, file_txt, stage)