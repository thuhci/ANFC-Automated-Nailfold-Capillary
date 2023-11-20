import numpy as np
import torch
import torchvision
import tqdm
from Keypoint_Detection.nailfold_keypoint.dataset import KeypointTestDataset
from Keypoint_Detection.nailfold_keypoint.rcnn_keypoint import \
    get_kp_rcnn_model
from Keypoint_Detection.nailfold_keypoint.rcnn_mask import get_mask_rcnn_model
from torch.utils.data import DataLoader


def t_images2kp_rcnn(imgs, batch_size=4, model_name="exp_tangshan_keypoint"):
    """
    Detect vascular keypoints in multiple images.

    Args:
        imgs (list of np.array): List of input images.
        batch_size (int, optional): Batch size for processing images. Defaults to 4.
        model_name (str, optional): Name of the keypoint detection model. Defaults to "exp_tangshan_keypoint".

    Returns:
        tuple: A tuple containing the following elements:
            - bboxes (list of np.array): List of bounding boxes for detected keypoints.
            - keypoints (list of np.array): List of detected keypoints.
            - scores (list of np.array): List of confidence scores for detected keypoints.
    """

    ############################## Train ################################
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    testset = KeypointTestDataset(imgs)

    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False) #TODO: #, collate_fn=collate_fn)

    model = get_kp_rcnn_model(model_name)  

    model.to(device)

    model.eval()

    iterator = tqdm.tqdm(test_loader)
    iterator.set_description("extract keypoints")

    all_keypoints = []
    all_bboxes = []

    for images, targets in iterator:
        images = list(image.to(device) for image in images)

        output = model(images)

        # images list[(3,400,600)]
        scores = [o['scores'].detach().cpu().numpy() for o in output]

        # Indexes of boxes with scores > 0.7
        high_scores_idxs = [np.where(score > 0.8)[0] for score in scores]
        post_nms_idxs =[torchvision.ops.nms(output[i]['boxes'][high_scores_idxs[i]], output[i]['scores'][high_scores_idxs[i]], 0.3).cpu().numpy()
                        for i in range(len(output))] # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

        for i in range(len(output)):
            keypoints = []
            for kps in output[i]['keypoints'][high_scores_idxs[i]][post_nms_idxs[i]].detach().cpu().numpy():
                keypoints.append([np.array(kp[:2]).astype(int) for kp in kps])

            bboxes = []
            for bbox in output[i]['boxes'][high_scores_idxs[i]][post_nms_idxs[i]].detach().cpu().numpy():
                bboxes.append(np.array(bbox).astype(int))
            
            all_keypoints.append(np.array(keypoints))
            all_bboxes.append(np.array(bboxes))

    return all_bboxes, all_keypoints, scores


def t_images2masks_rcnn(imgs, batch_size=4, model_name="exp_tangshan_segment"):
    """
    Perform instance segmentation on multiple images.

    Args:
        imgs (list of np.array): List of input images.
        batch_size (int, optional): Batch size for processing images. Defaults to 4.
        model_name (str, optional): Name of the segmentation model. Defaults to "exp_tangshan_segment".

    Returns:
        tuple: A tuple containing the following elements:
            - bboxes (list of np.array): List of bounding boxes for detected instances.
            - all_masks (list of np.array): List of segmentation masks for detected instances.
            - labels (list of np.array): List of labels for detected instances.
            - scores (list of np.array): List of confidence scores for detected instances.
    """

    ############################## Train ################################
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    testset = KeypointTestDataset(imgs)

    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False)#TODO:, collate_fn=collate_fn)
    
    model = get_mask_rcnn_model(model_name)
    model.to(device)

    model.eval()

    iterator = tqdm.tqdm(test_loader)
    iterator.set_description("extract instance masks")

    all_masks = []
    all_bboxes = []
    all_scores = []
    all_labels = []

    for images, targets in iterator:
        images = list(image.to(device) for image in images)

        output = model(images)

        # images list[(3,400,600)]
        scores = [o['scores'].detach().cpu().numpy() for o in output]

        # Indexes of boxes with scores > 0.5
        high_scores_idxs = [np.where(score > 0.5)[0] for score in scores]
        post_nms_idxs =[torchvision.ops.nms(output[i]['boxes'][high_scores_idxs[i]], output[i]['scores'][high_scores_idxs[i]], 0.3).cpu().numpy()
                        for i in range(len(output))] # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes

        for i in range(len(output)):
            masks = []
            # output[0] error Revised by lx
            for mask in output[i]['masks'][high_scores_idxs[i]][post_nms_idxs[i]].detach().cpu().numpy():
                masks.append((mask[0]>0.5).astype(np.bool8))

            bboxes = []
            for bbox in output[i]['boxes'][high_scores_idxs[i]][post_nms_idxs[i]].detach().cpu().numpy():
                bboxes.append(np.array(bbox).astype(int))
            labels = output[i]['labels'][high_scores_idxs[i]][post_nms_idxs[i]].detach().cpu().numpy()
            scores = output[i]['scores'][high_scores_idxs[i]][post_nms_idxs[i]].detach().cpu().numpy()
            all_labels.append(labels)
            all_scores.append(scores)
            all_masks.append(np.array(masks))
            all_bboxes.append(np.array(bboxes))

    return all_bboxes, all_masks, all_labels, all_scores

