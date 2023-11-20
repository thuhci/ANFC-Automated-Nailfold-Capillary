# evaluation 
### 1. Run full pipeline to auto measure all images in image_path by setting image_name to '':
`python Image_Analysis/nailfold_image_profile/overall_analysis.py --image_path "../Nailfold_Data_Tangshan/tangshan_data/tangshan_segmentation" --image_name '' --output_dir "./output_results"  --visualize`

the results will be saved in the output_dir.
got `./outputs_evaluation/labels/results_pred.json`

### 2. Analyze images from the same person and integrate them into the individual final report.
python ./eval/combine_results_pred_indivisual.py

`./outputs_evaluation/labels/results_pred_individual.json`

details:
- only images with visibility == 1 are regarded as valid images that are involved in the final caculation.
- caculating mean of the valid images for each person as final results.
- using NaN as the notion of invalid results.
- always check visibility before using individual results.

### 3. gt annotations from the doctors:
bool:
`./outputs_evaluation/labels/results_dict_gt_bool.json`

float results are from ptn_item file.
`./outputs_evaluation/labels/results_dict_gt_float.json`
details:
- manually revise certain annotation mistakes.
    - replacing 164.18 of top diameter with NaN.
    - missing values are filled with NaN.
    - swiching input and output diameter measurements if the ratio is < 1 

combine them into final results:
python ./eval/combine_results_gt.py

`./outputs_evaluation/labels/results_gt_individual.json`

### 4. compare gt and pred:
python ./eval/eval.py

results are saved at `./eval/results`

detailed:
- set r = 1.7 (r um/pixel) since it is about {np.mean(gt_labels)/np.mean(pred_labels)}