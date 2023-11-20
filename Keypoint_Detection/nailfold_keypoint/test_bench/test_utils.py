import numpy as np

def detect_acc(output, gt):
    """
    同一张照片的两组关键点，求其预测准确率等指标
    
    Args:
        output (np.array): (l, 1/6, 2)
        gt (np.array): (l', 1/6, 2)
    Returns:
        {"sensitivity": sensitivity,
        "precision": precision,
        "match_pos": np.array(l, 2)}
    """
    gt_match = np.zeros(gt.shape[0])
    output_match = np.zeros(output.shape[0])
    match_pos = []
    for i,gt_kp in enumerate(gt):
        for j,o_kp in enumerate(output):
            if np.abs(gt_kp[0]-o_kp[0]).sum() < 20:
                match_pos.append([i,j])
                gt_match[i] = 1
                output_match[j] = 1
    # gt中预测正确的比例
    sensitivity = (gt_match.sum() + 1e-6) / (gt_match.shape[0] + 1e-6)
    # 预测中正确的比例
    precision = (output_match.sum() + 1e-6) / (output_match.shape[0] + 1e-6)
    return {
        "sensitivity": sensitivity,
        "precision": precision,
        "match_pos": np.array(match_pos)
    }


def diameter_acc(output, gt, match_pos):
    """关键点管径的准确率

    Args:
        output (_type_): _description_
        gt (_type_): _description_
        match_pos (_type_): _description_
    Returns:
        {
        "top_error": top_error.mean(),
        "left_error": left_error.mean(),
        "right_error": right_error.mean()
        }
    """
    top_diameter_o = np.linalg.norm(output[:,0] - output[:,1], axis=-1)
    input_diameter_o = np.linalg.norm(output[:,2] - output[:,3], axis=-1)
    output_diameter_o = np.linalg.norm(output[:,4] - output[:,5], axis=-1)
    top_diameter_gt = np.linalg.norm(gt[:,0] - gt[:,1], axis=-1)
    input_diameter_gt = np.linalg.norm(gt[:,2] - gt[:,3], axis=-1)
    output_diameter_gt = np.linalg.norm(gt[:,4] - gt[:,5], axis=-1)
    top_error = abs(top_diameter_o[match_pos[:,1]] - top_diameter_gt[match_pos[:,0]]) / top_diameter_gt[match_pos[:,0]]
    left_error = abs(input_diameter_o[match_pos[:,1]] - input_diameter_gt[match_pos[:,0]]) / input_diameter_gt[match_pos[:,0]]
    right_error = abs(output_diameter_o[match_pos[:,1]] - output_diameter_gt[match_pos[:,0]]) / output_diameter_gt[match_pos[:,0]]
    return {
        "top_error": top_error,
        "left_error": left_error,
        "right_error": right_error
    }
    

