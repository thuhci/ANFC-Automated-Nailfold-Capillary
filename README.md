# Automated Nailfold Capillary Analysis  
 
This is a project for nailfold capillary automated analysis. The paper link:
[A Comprehensive Dataset and Automated Pipeline for Nailfold Capillary Analysis](https://arxiv.org/abs/2312.05930).
Nailfold capillaroscopy stands as a traditional and classical method for health condition assessment. However, manually employing this method poses challenges as the process of human measurement is not only time-consuming but also hindered by subjective criteria. In this research, we are pioneering the construction of a large dataset comprising 321 capillaroscopy images, 219 videos, and 68 clinic reports. This dataset includes annotations such as segmentations and keypoints from experts, forming a crucial resource for training deep-learning models. Leveraging our dataset, we introduce an end-to-end nailfold capillary analysis pipeline capable of automatically detecting and measuring a comprehensive set of morphological and dynamic features. The experiments demonstrate that our automated analysis algorithms achieve remarkable accuracy, which holds promise for quantitative medical research and pervasive computing in human health. We plan to open-source our datasets and code soon to facilitate
further study.

## 🔥 Updates
**[2024/2]** **Citation BibTex and Data Release Agreement are updated.**  
**[2023/11]** **Code is updated.**  

## 🔍 Data Sample  
![output](https://github.com/THU-CS-PI-LAB/ANFC-Automated-Nailfold-Capillary/assets/73820234/5fd17b34-20c8-46fe-8615-ae0805caaba3)
<div style="text-align: center;">
    <img src="https://github.com/THU-CS-PI-LAB/ANFC-Automated-Nailfold-Capillary/blob/main/demo.gif" style="width: 50%; height: auto;">
</div>


## 🖥️ Dataset Structure
```
ANFC_THU
├── ANFC_THU_data
 ├── ANFC_THU_keypoint
  ├── SubjectID_PicID.jpg   #raw data
  ├── SubjectID_PicID.json  #label
 ├── ANFC_THU_segmentation
  ├── SubjectID_PicID.jpg   #raw data
  ├── SubjectID_PicID.json  #label
```
## 🗝️ Access and Usage
**This dataset is built for academic use. Any commercial usage is banned.**  
There are two ways for downloads： OneDrive and Baidu Netdisk for researchers of different regions.  
To access the dataset, you are supposed to download this [data release agreement](https://github.com/THU-CS-PI-LAB/ANFC-Automated-Nailfold-Capillary/blob/main/ANFC_THU_Release_Agreement.pdf).  
Please scan and dispatch the completed agreement via your institutional email to <tjk19@mails.tsinghua.edu.cn> and cc <yuntaowang@tsinghua.edu.cn>. The email should have the subject line 'ANFC_THU Access Request -  your institution.' In the email,  outline your institution's past research and articulate the rationale for seeking access to the ANFC_THU, including its intended application in your specific research project.   

## ⚙️ Setup

STEP1: `bash setup.sh` 

STEP2: `conda activate nailfold` 

STEP3: `pip install -r requirements.txt` 

## ⚙️ Run Pipeline
### Run Image Automated Analysis Pipeline
- Run full pipeline for specific image analysis:
` python Image_Analysis/nailfold_image_profile/overall_analysis.py`

- full command:
` python Image_Analysis/nailfold_image_profile/overall_analysis.py --image_path "../Nailfold_Data_Tangshan/tangshan_data/tangshan_segmentation" --image_name "8_58452_5.jpg" --output_dir "./output_test" --visualize`

- Run full pipeline for all images in image_path, just set image_name to '':
`python Image_Analysis/nailfold_image_profile/overall_analysis.py --image_path "../Nailfold_Data_Tangshan/tangshan_data/tangshan_segmentation" --image_name '' --output_dir "./output_results"  --visualize`

### Run Video Automated Analysis Pipeline
- Analyze the all videos in $video_path_dict_file$ and return the velocity of the white blood cell:
`python Flow_Velocity_Measurement/video_profile.py  --video_type ".mp4" --video_path ./Flow_Velocity_Measurement/video_sample --output_dir ./Flow_Velocity_Measurement/output_table/ --nailfold_pos_x 150 --nailfold_pos_y 100 --split_num 1 --pad_ratio 2 --video_path_dict_file ./outputs/aligned_video_path_dict.json --visualize`

### Video Profiles(WBC Count and Flow Velocity Measurement)
Analyze the video and return the velocity of the white blood cell:
- video example
`python Flow_Velocity_Measurement/video_profile.py --video_name "kp-6" --video_type ".mp4" --video_path ./Flow_Velocity_Measurement/video_sample --output_dir ./Flow_Velocity_Measurement/output/ --nailfold_pos_x 150 --nailfold_pos_y 100 --visualize --split_num 1 --pad_ratio 2`

---
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

- Training Log at
`./Image_Segmentation/image_segmentation/result/U_Net-60-0.0002-70-0.7000.log`

- Training Results Visualization at
`./Image_Segmentation/image_segmentation/result/visualization/visualize_pred_gt`

4. Test the model

```
python Image_Segmentation/image_segmentation/main.py  --mode=test --num_epochs=60 --val_step=5
```
---

### RCNN for Keypoint Detection
interface:
`from Keypoint_Detection.nailfold_keypoint.detect_rcnn import t_images2kp_rcnn, t_images2masks_rcnn`

Please use config files under ``

Dataset
`./Keypoint_Detection/data/nailfold_dataset_crossing`
`./Keypoint_Detection/data/nailfold_dataset1`

Train the model
`./Keypoint_Detection/nailfold_keypoint`

Test the model
``
### Resnet18 for Classifier (normal or abnormal)
TBD


---
## Toolbox Interface
### for image segmentation
`from Image_Segmentation.image_segmentation.image2segment import t_images2masks`

### for image keypoint detection and instance segmentation
`from Keypoint_Detection.nailfold_keypoint.detect_rcnn import t_images2kp_rcnn, t_images2masks_rcnn`

### for video analysis
Analyze the video and return the velocity of the white blood cell:
`t_video_analysis(video_path, output_dir, pos: tuple, visualize: bool = False, split_num: int = 2, pad_ratio: float= 1)->typing.List[float]`


---
## Dataset Dir
1. original dataset
./data

2. all dataset used in each tasks
./Data_Preprocess/data

3. video frame dataset
./Data_Preprocess/data/new_data_frame

4. keypoints dataset
original data patch:
./Keypoint_Detection/data
original dataset for all:
./Keypoint_Detection/data/nailfold_dataset1
original dataset for crossing point
./Keypoint_Detection/data/nailfold_dataset_crossing

5. classifer
./Object_Detection/data

6. videos
./Video_Process/data
/kp_videos for stabilized videos
/videos are original one

---
## checkpoints Dir
1. Segmentation model
./Image_Segmentation/image_segmentation/checkpoints/U_Net-60-0.0002-70-0.7000.pkl

2. keypoints
./Keypoint_Detection/nailfold_keypoint/checkpoints

3. classifier
./Object_Detection/nailfold_classifier/checkpoints

## Citation  
```
@misc{zhao2023comprehensive,
    title={A Comprehensive Dataset and Automated Pipeline for Nailfold Capillary Analysis},
    author={Linxi Zhao and Jiankai Tang and Dongyu Chen and Xiaohong Liu and Yong Zhou and Guangyu Wang and Yuntao Wang},
    year={2023},
    eprint={2312.05930},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
