import math
import os
import time

import Keypoint_Detection.nailfold_keypoint.utils.utils as utils
import torch
import torchvision.models.detection.mask_rcnn
from Keypoint_Detection.nailfold_keypoint.coco_eval import CocoEvaluator
from Keypoint_Detection.nailfold_keypoint.coco_utils import \
    get_coco_api_from_dataset
from torch.optim.lr_scheduler import LinearLR


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # print(loss_dict_reduced)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)
        for k, v in loss_dict_reduced.items():
            # print(k, v.item())
            writer.add_scalar("Train/"+k, v.item(), global_step=epoch)

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, keypoint_num):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, keypoints_number=keypoint_num)

    for images, targets in metric_logger.log_every(data_loader, 5, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # for k, v in loss_dict_reduced.items():
        #     writer.add_scalar("Test/"+k, v.item())

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def _get_state_dict(model_name):
    cur_path=os.path.abspath(__file__)
    cur_path = os.path.dirname(cur_path)
    ckpt_path = f'checkpoints/{model_name}/ckpts/rcnn_weights_epo_49.pth'
    state_dict = torch.load(os.path.join(cur_path,ckpt_path))
    class_num = 0
    keypoint_num = 0
    if "class_num" in state_dict:
        class_num = state_dict["class_num"]
        del state_dict["class_num"]
    if "keypoint_num" in state_dict:
        keypoint_num = state_dict["keypoint_num"]
        del state_dict["keypoint_num"]
    return state_dict, keypoint_num, class_num