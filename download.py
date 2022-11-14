import gdown
import os

project_path = '/storage/hieunmt/zaloai_liveness'

try:
    os.mkdir(project_path)
except:
    pass

des = os.path.join(project_path, "download")

try:
    os.mkdir(des)
except:
    pass

# url = "https://drive.google.com/file/d/1wZxjvJiCUPpl5OjfwKxJRvLdbS1xmusr/view?usp=share_link"
# output = f"{des}/ViT-B-16.pt"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1z-vHZtAAyZ_WmZfVsa7LBHUwCnW6Fx0E/view?usp=share_link"
# output = f"{des}/train.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1yIAjyr-kyurS3qD7YuuobhoixoZI9QBv/view?usp=share_link"
# output = f"{des}/public_test.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1dy5Qy0xp6XnpLOYf3JBVODKNPS_SADVD/view?usp=share_link"
output = f"{des}/public_test_2.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1XBXm6gwQ88zCqXqCtQUrniVQWAR3wM4X/view?usp=share_link"
# output = f"{des}/fold_0.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/17eMGrbXjYPTv2FstOQNP8B7mhlh392Pw/view?usp=share_link"
# output = f"{des}/seed_1024_20_fold_0.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1dax4qUIOEI_QzYXv31J-87cDkonQetVQ/view?usp=sharing"
# output = f"{des}/k400_vitb16_16f_dec4x768.pth"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/19T_NN3baGqSlRv9t4I650oA9B0_81JXg/view?usp=share_link"
# output = f"{des}/seed_128_10_fold_0.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1CTV9geLD3HLWzByAQUOf_m0F_g2lE3rg/view?usp=sharing"
# output = f"{des}/k400_vitl14_16f_dec4x1024.pth"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1fGLhe1sOMo0OuFn3qC4qjn4CohJeJ22a/view?usp=share_link"
# output = f"{des}/ViT-L-14.pt"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/19yJwlVUZyiL-3aaevjmLklsxHooWjsBD/view?usp=share_link"
# output = f"{des}/seed_42_20_fold_10.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)