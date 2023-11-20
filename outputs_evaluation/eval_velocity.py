import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predicted labels against ground truth labels.

    Parameters:
    - y_true: List or array of ground truth labels.
    - y_pred: List or array of predicted labels.

    Returns:
    - Dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {}

    # Calculate metrics
    metrics['accuracy'] = round(accuracy_score(y_true, y_pred),3)
    metrics['precision'] = round(precision_score(y_true, y_pred, average='weighted'),3)
    metrics['recall'] = round(recall_score(y_true, y_pred, average='weighted'),3)
    metrics['f1_score'] = round(f1_score(y_true, y_pred, average='weighted'),3)
    metrics["sample size"] = len(gt_labels)

    return metrics


def get_valid_labels_pair(gt, pred, key):
    pred_keys_ls = list(pred.keys())
    gt_keys_ls = list(gt.keys())
    pred_keys_ls.sort()
    gt_keys_ls.sort()
    assert pred_keys_ls == gt_keys_ls
    
    gt_labels = np.array([gt[k]['image_info'][key] for k in pred_keys_ls])
    pred_labels = np.array([pred[k]['image_info'][key] for k in pred_keys_ls])
    pred_visibility = np.array([pred[k]['image_info']['visibility'] for k in pred_keys_ls])
    gt_visibility = np.array([gt[k]['image_info']['visibility'] for k in pred_keys_ls])
    # filter nan data
    nan_idx = np.isnan(gt_labels) + np.isnan(pred_labels)
    # filter those "visibility" is 0
    # delete error
    invalid_idx = nan_idx + (gt_visibility == 0) + (pred_visibility == 0) + (gt_labels < 0) + (pred_labels < 0)
    gt_labels = gt_labels[~invalid_idx]
    pred_labels = pred_labels[~invalid_idx]

    return gt_labels, pred_labels


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def draw_scatter(gt_labels, pred_labels, feature, save_dir):
    x = gt_labels
    y = pred_labels
    
    y_clean = y[~np.isnan(y) & ~np.isnan(x)]
    x_clean = x[~np.isnan(y) & ~np.isnan(x)]

    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean,X).fit()
    print(feature, model.summary())

    # draw regression plot and residual plot
    plt.figure(figsize=(16,8))

    ax1 = plt.subplot(1, 2, 1)
    sns.regplot(x=x_clean,y=y_clean,label = f"y={round(model.params[1],2)}x+{round(model.params[0],2)} R2={round(model.rsquared,3)}")
    x1 = np.linspace(min(x),max(x),100)
    ax1.plot(x1,x1,ls='--',c='k')

    ax2 = plt.subplot(1, 2, 2)
    sns.residplot(x=x_clean,y=y_clean, lowess=True, line_kws=dict(color="r"))
    
    ax1.set_xlabel("gt")
    ax1.set_ylabel("pred")
    ax2.set_xlabel("gt")
    ax2.set_ylabel("residual")
    ax1.legend()
    ax1.set_title(f"{feature} - regression plot")
    ax2.set_title(f"{feature} - residual plot")

    save_fig = os.path.join(save_dir,f"eval_{feature.replace(' ', '_')}.png")
    plt.savefig(save_fig)
    print(f"Save fig at {save_fig}")
    plt.close()

def evaluate_lengths(gt_lengths, pred_lengths):
    """
    Evaluate predicted lengths against ground truth lengths.

    Parameters:
    - gt_lengths: List or array of ground truth lengths.
    - pred_lengths: List or array of predicted lengths.

    Returns:
    - Dictionary containing mean absolute error (MAE) and mean squared error (MSE).
    """
    metrics = {}

    # Calculate metrics
    metrics['mae'] = round(mean_absolute_error(gt_lengths, pred_lengths),3)
    metrics['mse'] = round(mean_squared_error(gt_lengths, pred_lengths),3)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='./outputs_evaluation/labels')
    parser.add_argument('--pred_dir', type=str, default='./Flow_Velocity_Measurement/output_test')
    parser.add_argument('--pred_file', type=str, default='results_velocity_individual.json')
    parser.add_argument('--gt_file', type=str, default='results_dict_gt_velocity.json')
    parser.add_argument('--save_dir', type=str, default='./outputs_evaluation/results')
    parser.add_argument('--save_file', type=str, default='eval_velocity.json')
    parser.add_argument('--save_fig', type=str, default='eval_velocity.png')
    args = parser.parse_args()


    path_gt = os.path.join(args.gt_dir, args.gt_file)
    path_pred = os.path.join(args.pred_dir, args.pred_file)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_file)
    save_fig_path = os.path.join(args.save_dir, args.save_fig)
    
    with open(path_gt, 'r') as f:
        gt = json.load(f)

    with open(path_pred, 'r') as f:
        pred = json.load(f)
    
    pred_dict = {}
    for item in pred:
        pred_dict.update(item)
    # print(pred_dict)
        
    metrics_dict = {}
    
    ## float
    results_dict = {}
    fig = plt.figure(figsize=(16,4))

    pred_keys_ls = list(pred_dict.keys())
    gt_keys_ls = list(gt.keys())
    gt_labels, pred_labels = [], []
    num_wbc = []
    for key in pred_keys_ls:
        if key in gt_keys_ls:
            gt_labels.append(gt[key])
            pred_labels.append(pred_dict[key]['velocity'])
            num_wbc.append(len(pred_dict[key]['wbc_events']))
    
    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)
    nan_idx = np.isnan(gt_labels) + np.isnan(pred_labels)
    # filter those "visibility" is 0
    # delete error
    
    invalid_idx = nan_idx + (gt_labels < 0) + (pred_labels < 0) + (num_wbc) < 2 
    gt_labels = gt_labels[~invalid_idx]
    pred_labels = pred_labels[~invalid_idx]

    results_lengths = evaluate_lengths(gt_labels, pred_labels)
    results_dict[key] = results_lengths
    
    gt_labels_bool = gt_labels >= 600
    pred_labels_bool = pred_labels >= 600
    results = evaluate_predictions(gt_labels_bool, pred_labels_bool)
    results_dict[key+'_bool'] = results

    # draw scatter
    draw_scatter(gt_labels, pred_labels, key, args.save_dir)

        # draw scatter with color
        # tp = gt_labels_bool & pred_labels_bool # 3
        # fp = ~gt_labels_bool & pred_labels_bool # 9
        # fn = gt_labels_bool & ~pred_labels_bool # 6
        # tn = ~gt_labels_bool & ~pred_labels_bool # 0
        # z = 3*np.array(tp).astype(np.int) + 9*np.array(fp).astype(np.int) + 6*np.array(fn).astype(np.int)

        # # do not show red dashed line
        # ax = fig.add_subplot(1, 4, ft_keys_ls.index(key)+1)
        # ax.scatter(gt_labels, pred_labels, s=20, c=z, cmap='coolwarm', alpha=0.8, marker='o', label=f"MAE={results_lengths['mae']}, MSE={results_lengths['mse']}")
        # # ax.plot([0.8*min(gt_labels), 1.2*max(gt_labels)], [0.8*min(pred_labels), 1.2*max(pred_labels)], 'r--')
        # ax.set_xlim([0.8 * min(gt_labels), 1.2 * max(gt_labels)])
        # ax.set_ylim([0.8 * min(pred_labels), 1.2 * max(pred_labels)])

        # ax.set_xlabel('ground truth / um')
        # ax.set_ylabel('prediction / pixel')
        # ax.set_title(key)
        # ax.legend()
        # annotate mae and mse in the plot
        

    # save plt
    # plt.tight_layout()
    # fig.savefig(save_fig_path, dpi=300)
    # plt.close(fig)
    # print(f"Save figure at {save_fig_path}")

    ## bool
    # for key in bool_keys_ls:
    #     gt_labels, pred_labels = get_valid_labels_pair(gt, pred, key)
        
    #     results = evaluate_predictions(gt_labels, pred_labels)
    #     results_dict[key] = results

    save_dict = {"gt": path_gt, "pred": path_pred, "eval results": results_dict}
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=4)
        print("Evaluation results saved in {}".format(save_path))