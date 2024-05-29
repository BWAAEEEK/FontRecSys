import os
from transformers import ViTImageProcessor
from PIL import Image
from tqdm import tqdm


def image_processing(processor: ViTImageProcessor):
    ppt_list = os.listdir("../Data/data_collection/MangoCrawling/output/images")

    ppt_image = {name: [] for name in ppt_list}

    for name in tqdm(ppt_list, desc="image processing"):
        slide_list = os.listdir(f"../Data/data_collection/MangoCrawling/output/images/{name}")

        for slide in slide_list:
            image = Image.open(f"../Data/data_collection/MangoCrawling/output/images/{name}/{slide}")
            image = image.resize((1024, 1024))

            image = processor(image, return_tensors="pt", size={"height": 1024, "width": 1024})

            ppt_image[name].append(image)

    return ppt_image

