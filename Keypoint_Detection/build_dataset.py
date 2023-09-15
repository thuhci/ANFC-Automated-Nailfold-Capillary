import os
import json
import tqdm

with open("/data3/yingke/PI-NVC-Tangshan-Samples/images_path.json","r") as f:
    image_json = json.load(f)

image_path = image_json['images']
base = "./tangshan/images"
os.makedirs(base,exist_ok=True)
for i,img in tqdm.tqdm(enumerate(image_path)):
    dest = os.path.join(base,str(i))
    os.system(f'cp {img} {dest}.png')
