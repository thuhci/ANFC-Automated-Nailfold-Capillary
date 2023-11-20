import tqdm
from Keypoint_Detection.nailfold_keypoint.main import *

KEYPOINTS_FOLDER_TRAIN = './data/data/nailfold_dataset_crossing/train'

dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, demo=True)
data_loader = DataLoader(dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_fn)

for id,batch in tqdm.tqdm(enumerate(data_loader)):
    image = (batch[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bboxes = batch[1][0]['boxes'].numpy().astype(np.int32)

    keypoints = []
    for kps in batch[1][0]['keypoints'].numpy().astype(np.int32):
        keypoints.append([kp[:2] for kp in kps])

    labels = batch[1][0]['labels'].numpy().astype(np.int32)

    visualize(f"{id}", image, bboxes, keypoints, labels=labels, output_dir="./data/visualize")