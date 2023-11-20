import os

import cv2
import numpy as np
import tqdm
from Keypoint_Detection.nailfold_keypoint.dataset import MaskDataset_RCNN
from Keypoint_Detection.nailfold_keypoint.utils.utils import collate_fn
from Keypoint_Detection.nailfold_keypoint.utils.utils_visualize import \
    draw_mask_on_image
from torch.utils.data import DataLoader

DATASET = "./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/train"
output = "./data/segment_visualize"

os.makedirs(output, exist_ok=True)
dataset = MaskDataset_RCNN(DATASET)
data_loader = DataLoader(dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_fn)

for id,batch in tqdm.tqdm(enumerate(data_loader)):
    image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, masks, labels = dataset.target2numpy(batch[1])
    vis = draw_mask_on_image(image, bboxes, masks, None, labels.tolist())
    name = dataset.imgs_files[int(batch[1][0]['image_id'].item())]
    cv2.imwrite(os.path.join(output, name), vis)

