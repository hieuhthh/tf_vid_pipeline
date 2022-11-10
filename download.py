import gdown
import os

project_path = '/home/lap14880/hieunmt/tf_vid_pipeline'

try:
    os.mkdir(project_path)
except:
    pass

des = os.path.join(project_path, "download")

try:
    os.mkdir(des)
except:
    pass

url = "https://drive.google.com/file/d/1wZxjvJiCUPpl5OjfwKxJRvLdbS1xmusr/view?usp=share_link"
output = f"{des}/ViT-B-16.pt"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1z-vHZtAAyZ_WmZfVsa7LBHUwCnW6Fx0E/view?usp=share_link"
output = f"{des}/train.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1yIAjyr-kyurS3qD7YuuobhoixoZI9QBv/view?usp=share_link"
output = f"{des}/public_test.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1XBXm6gwQ88zCqXqCtQUrniVQWAR3wM4X/view?usp=share_link"
# output = f"{des}/fold_0.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/17eMGrbXjYPTv2FstOQNP8B7mhlh392Pw/view?usp=share_link"
output = f"{des}/seed_1024_20_fold_0.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)