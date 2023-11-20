# github
git remote add origin_nailfold git@lx.github.com:Linxi-ZHAO/nailfold.git

git push -u origin_nailfold main


# Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate nailfoldtoolbox` 

STEP3: `pip install -r requirements.txt` 

# Run the code
`conda activate pytorch`

`cd Full_Pipeline`

## Pipeline Example
Run full pipeline for specific image analysis:
` python Image_Analysis/nailfold_image_profile/overall_analysis.py`

` python Image_Analysis/nailfold_image_profile/overall_analysis.py --image_path "../Nailfold_Data_Tangshan/tangshan_data/tangshan_segmentation" --image_name "8_58452_5.jpg" --output_dir "./output_test" --visualize`
Run full pipeline for all images in image_path, just set image_name to '':
`python Image_Analysis/nailfold_image_profile/overall_analysis.py --image_path "../Nailfold_Data_Tangshan/tangshan_data/tangshan_segmentation" --image_name '' --output_dir "./output_results"  --visualize`

Analyze the video and return the velocity of the white blood cell:
`python Flow_Velocity_Measurement/video_profile.py --video_name "kp-6" --video_type ".mp4" --video_path ./Flow_Velocity_Measurement/video_sample --output_dir ./Flow_Velocity_Measurement/output/ --nailfold_pos_x 150 --nailfold_pos_y 100 --visualize --split_num 1 --pad_ratio 2`

## Models
### UNet for segmetation
1. interface:
`from Image_Segmentation.image_segmentation.image2segment import t_images2masks`

2. Please use config files under `Image_Segmentation/image_segmentation/config.yaml`

3. Train the model
- brief command
```
python Image_Segmentation/image_segmentation/main.py  --mode=train --num_epochs=60 --val_step=5 --batch_size=8
```

- whole command
```
python Image_Segmentation/image_segmentation/main.py  --mode=train 
--image_size=256 --img_ch=1 --output_ch=1 --batch_size=4 --num_workers=8 --augmentation_prob=0.7 --num_epochs=60 --num_epochs_decay=70 --lr=0.0002 --beta1=0.5 --beta2=0.999 --val_step=2 --log_step=2 --model_type=U_Net --model_path=./Image_Segmentation/image_segmentation/models --train_path=./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/train --valid_path=./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/test --test_path=./Data_Preprocess/nailfold_data_preprocess/data/segment_dataset/test --result_path=./Image_Segmentation/image_segmentation/result --cuda_idx=1
```
- epoch=60, lr = 0.0002, training result is 
`[hightlight] Test Acc: 0.9616, SE: 0.6532, F1: 0.4871`

```
[Training] Acc: 0.9644, SE: 0.7290, SP: 0.9709, PC: 0.3894, F1: 0.5008, JS: 0.3355, DC: 0.5008
[Test] Acc: 0.9616, SE: 0.6532, SP: 0.9706, PC: 0.3918, F1: 0.4871, JS: 0.3229, DC: 0.4871

```
- Pretrained Model at
```
Best U_Net model score : 0.4811 at epoch: 45
save at /home/user/nailfold/zhaolx/Full_Pipeline/Image_Segmentation/image_segmentation/models/U_Net-60-0.0002-70-0.7000.pkl
```

- Training Log at
`/home/user/nailfold/zhaolx/Full_Pipeline/Image_Segmentation/image_segmentation/result/U_Net-60-0.0002-70-0.7000.log`

- Training Results Visualization at
`/home/user/nailfold/zhaolx/Full_Pipeline/Image_Segmentation/image_segmentation/result/visualization/visualize_pred_gt`

4. Test the model

```
python Image_Segmentation/image_segmentation/main.py  --mode=test --num_epochs=60 --val_step=5
```
---
### RCNN for Keypoint Detection
interface:
`from Keypoint_Detection.nailfold_keypoint.detect_rcnn import t_images2kp_rcnn, t_images2masks_rcnn`
(注： interface都修改为end2end的函数接口，命名统一为t_xxx，t意为tool)
（注： detect_imgs为关键点检测，通过参数model_name来区分是顶端关键点还是交叉点模型）
Please use config files under `暂无`

Dataset
`/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/data/nailfold_dataset_crossing`
Train the model
`/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/nailfold_keypoint 文件夹下`

Test the model
``

### Video Profiles(WBC Count and Flow Velocity Measurement)
Analyze the video and return the velocity of the white blood cell:
- video example
`python Flow_Velocity_Measurement/video_profile.py --video_name "kp-6" --video_type ".mp4" --video_path ./Flow_Velocity_Measurement/video_sample --output_dir ./Flow_Velocity_Measurement/output/ --nailfold_pos_x 150 --nailfold_pos_y 100 --visualize --split_num 1 --pad_ratio 2`
`CUDA_VISIBLE_DEVICES=1 python Flow_Velocity_Measurement/video_profile.py --video_name "115825" --video_type ".mp4" --video_path ./Flow_Velocity_Measurement/video_sample --output_dir ./Flow_Velocity_Measurement/output_test/ --nailfold_pos_x 150 --nailfold_pos_y 100 --visualize --split_num 1 --pad_ratio 1 --video_path_dict_file ./outputs/aligned_video_path_dict.json`

CUDA_VISIBLE_DEVICES=1 python Flow_Velocity_Measurement/video_profile.py  --video_type ".mp4" --video_path ./Flow_Velocity_Measurement/video_sample --output_dir ./Flow_Velocity_Measurement/output_table/ --nailfold_pos_x 150 --nailfold_pos_y 100 --split_num 1 --pad_ratio 2 --video_path_dict_file ./outputs/aligned_video_path_dict.json --visualize

---
# Toolbox Interface
## for image segmentation
`from Image_Segmentation.image_segmentation.image2segment import t_images2masks`

## for image keypoint detection and instance segmentation
`from Keypoint_Detection.nailfold_keypoint.detect_rcnn import t_images2kp_rcnn, t_images2masks_rcnn`

## for video analysis
Analyze the video and return the velocity of the white blood cell:
`t_video_analysis(video_path, output_dir, pos: tuple, visualize: bool = False, split_num: int = 2, pad_ratio: float= 1)->typing.List[float]`


---
### Dataset
1. original dataset
/home/user/nailfold/zhaolx/Full_Pipeline/data

2. all dataset used in each tasks
/home/user/nailfold/zhaolx/Full_Pipeline/Data_Preprocess/data

3. video frame dataset
/home/user/nailfold/zhaolx/Full_Pipeline/Data_Preprocess/data/new_data_frame

4. keypoints dataset
original data patch:
/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/data
original dataset for all:
/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/data/nailfold_dataset1
original dataset for crossing point
/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/data/nailfold_dataset_crossing

5. classifer
/home/user/nailfold/zhaolx/Full_Pipeline/Object_Detection/data

6. videos
/home/user/nailfold/zhaolx/Full_Pipeline/Video_Process/data
/kp_videos for stabilized videos
/videos are original one

### checkpoints
1. Segmentation model
/home/user/nailfold/zhaolx/Full_Pipeline/Image_Segmentation/image_segmentation/checkpoints/U_Net-60-0.0002-70-0.7000.pkl

2. keypoints
/home/user/nailfold/zhaolx/Full_Pipeline/Keypoint_Detection/nailfold_keypoint/checkpoints

3. classifier
/home/user/nailfold/zhaolx/Full_Pipeline/Object_Detection/nailfold_classifier/checkpoints