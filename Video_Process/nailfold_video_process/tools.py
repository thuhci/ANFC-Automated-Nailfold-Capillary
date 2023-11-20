from Video_Process.nailfold_video_process.utils.transform import TangshanKp2Zhuhai, Tangshan2Zhuhai, TangshanCrop2Zhuhai
from Image_Segmentation.image_segmentation.image2segment import t_images2masks
from Keypoint_Detection.nailfold_keypoint.utils.utils_visualize import draw_kps_on_image
from Keypoint_Detection.nailfold_keypoint.detect_rcnn import t_images2kp_rcnn
import numpy as np
import tqdm
import cv2
import os

class NailfoldTools():
    def __init__(self) -> None:
        pass

    def img2segment(self,sample):
        # transform = Tangshan2Zhuhai()
        # samples = transform.transform(sample)
        # segmentation
        masks = t_images2masks([sample])
        # mask = transform.mask_rev_transform(masks.astype(np.bool8))
        mask = masks[0].astype(np.uint8) * 255
        return mask

    def img2keypoint(self,sample):
        # transform = Tangshan2Zhuhai()
        # samples = transform.transform(sample)
        # keypoint detection
        bboxes, keypoints, scores = t_images2kp_rcnn([sample])
        # keypoints = transform.kp_rev_transform(keypoints)
        # bboxes = transform.bbox_rev_transform(bboxes)
        return keypoints[0], bboxes[0]

    def imgs2segments(self,imgs):
        """
        将视频全部输出为segmentation
        """
        # segments = []
        # transforms = []
        # all_samples = []
        # start_list = [0]
        # for img in tqdm.tqdm(imgs):
        #     transform = Tangshan2Zhuhai()
        #     samples = transform.transform(img)
        #     transforms.append(transform)
        #     all_samples += samples
        #     start_list.append(len(all_samples))
        # segmentation
        all_masks = t_images2masks(imgs, 1)
        # for i,img in enumerate(imgs):
        #     transform = transforms[i]
        #     masks = all_masks[start_list[i]:start_list[i+1]]
        #     mask = transform.mask_rev_transform(masks.astype(np.bool8))
        #     mask = mask.astype(np.uint8) * 255
        #     segments.append(mask)
        return 255*all_masks

    def imgs2keypoint(self,imgs):
        """一组照片列表转化为其对应的关键点

        Args:
            imgs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # t = Tangshan2Zhuhai()
        # samples = []
        # for img in imgs:
        #     samples += t.transform(img)
        all_bboxes, all_keypoints, all_scores = t_images2kp_rcnn(imgs, batch_size=16)
        # all_keypoints = [t.kp_rev_transform(all_keypoints[i*t.get_size():(i+1)*t.get_size()]) for i in range(int(len(all_keypoints)/t.get_size()))]
        # all_bboxes = [t.bbox_rev_transform(all_bboxes[i*t.get_size():(i+1)*t.get_size()]) for i in range(int(len(all_bboxes)/t.get_size()))]
        return all_keypoints, all_bboxes

    def part_imgs2keypoint(self,imgs):
        """一组照片列表的截取中央区域转化为其对应的关键点

        Args:
            imgs (_type_): _description_

        Returns:
            _type_: _description_
        """
        t = TangshanCrop2Zhuhai()
        samples = [ t.transform(img) for img in imgs ]
        all_bboxes, all_keypoints, all_scores = t_images2kp_rcnn(samples, batch_size=16)
        all_keypoints = [t.kp_rev_transform(kps) for kps in all_keypoints]
        all_bboxes = [t.kp_rev_transform(kps) for kps in all_bboxes]
        return all_keypoints, all_bboxes

    def visualize_keypoint(self, name, image, bboxes, keypoints, scores=None, output_dir="./"):
        vis = draw_kps_on_image(input=image, bboxes=bboxes, keypoints=keypoints, scores=scores)
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), vis)



    def kpimage2segment(self, input):
        transform = TangshanKp2Zhuhai()
        input = transform.transform(input.astype(np.uint8))
        ori_segment = 255*t_images2masks([input], scale=1)[0]
        ori_segment = transform.rev_transform(ori_segment)
        return ori_segment

    def kpimgs2segments(self, imgs, use_batch=True):
        t = TangshanKp2Zhuhai()
        transform_imgs = [t.transform(img) for img in imgs]
        # if use_batch:
        #     segments = imgs2segments(np.array(transform_imgs))
        # else:
        #     it = tqdm.tqdm(enumerate(transform_imgs))
        #     it.set_description("Segmenting")
        #     for i,img in it:
        #         seg = t_images2masks(img, scale=1)
        #         segments.append(seg)
        segments = 255*t_images2masks(np.array(transform_imgs), scale=1)
        segments = [t.rev_transform(seg) for seg in segments]
        return segments