# Video Process

### Installation

```bash
# test in python 3.8
pip install opencv-python # 4.5.5.62 was tested
```

### Analysis pipeline
```bash
python stablize_all.py --input /home/user/nailfold/Tangshan-Samples/7.28 --output ./videos/7.28 --type avi

python stablize_all.py --input ./videos/7.28 --output ./Video_Process/data/aligned_videos/7.28 --type stable

python analysis_all.py --output ./kp_videos/7.28 --input ./Video_Process/data/aligned_videos/7.28 
```

### Stablize Video

* rouphly stablize a nailfold video by SIFT

```
# input: origin video's directory
# output: output directory
python video2video.py --input /data3/yingke/PI-NVC-Tangshan-Samples/7.28/49510 --output ./Video_Process/data/aligned_videos/49512
```

### Devide into Keypoint Video

* Use Nailfold Keypoint Detection to devide a large video into small videos with a nailfold capillary at the centre

```
# input: origin video's PATH
# output: output kp videos directory
python video2keypoint.py --input ./Video_Process/data/aligned_videos/49511/wmv1.mp4 --output ./Video_Process/data/aligned_videos/49511/wmv1
```

### Accurate Stablize Video

* Use Nailfold Segmentation and Phase-Correlation Method, to stablize a keypoint video accurately, assuring pixel level registration

```
# input: kp videos' directory
# output: output aligned video directory
python phase_correlation.py --input ./Video_Process/data/aligned_videos/59986/wmv2 --output ./Video_Process/data/aligned_videos/59986/wmv2-aligned
```

### Estimate Flow

* Use traditional image process method to extract white cells and calculate flow speed

```
# input: kp videos' directory
# output: output flow video directory
python estimate_flow.py --input ./Video_Process/data/aligned_videos/59986/wmv2-aligned --output ./Video_Process/data/aligned_videos/59986/wmv2-aligned
```
