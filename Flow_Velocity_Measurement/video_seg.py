import cv2
import numpy as np
from Image_Segmentation.image_segmentation.image2segment import t_images2masks


def process_video(i_video, num):
    cap = cv2.VideoCapture(i_video)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) >= 3:
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    if not cap.isOpened():
        print("Please check the path.")
    cnt = 0
    count = 0
    imgs = []
    while 1:
        ret, frame = cap.read()
        cnt += 1
        if cnt % num == 0:
            count += 1
            imgs.append(frame)

        if not ret:
            break
    cap.release()
    return imgs[:-1], fps


def get_video_seg(video_path, interval=1):
    '''
    This function is used to get the segmentation results of the video.
    '''
    # 抽帧
    images, fps = process_video(video_path, interval)
    new_imgs = []
    for id_s, img in enumerate(images[:]):
        img = img.astype(np.uint8)  # 480,640,3
        new_imgs.append(img)
    new_imgs = np.array(new_imgs)
    return t_images2masks(new_imgs, 1)


def get_video_best_seg(images, split_num=2, pad_ratio=1):
    '''
    This function is used to get the best segmentation result of the video.
    '''
    mask_score = np.zeros(len(images))
    new_images = []
    for id_s, img in enumerate(images[:]):
        img = img.astype(np.uint8)  # 480,640,3
        new_images.append(img[:, :, :])
    masks = t_images2masks(new_images, split_num, pad_ratio)  # (149, 300, 200)
    mask_score = np.sum(np.sum(masks, axis=1), axis=1)
    best_seg = masks[np.argmax(mask_score), :, :]

    # use the intersection of the best k segs as the final seg
    top_k = 3
    top_k_idx = mask_score.argsort()[::-1][0:top_k+1]
    for i in range(top_k):
        best_seg = masks[top_k_idx[i], :, :] | best_seg

    return best_seg
