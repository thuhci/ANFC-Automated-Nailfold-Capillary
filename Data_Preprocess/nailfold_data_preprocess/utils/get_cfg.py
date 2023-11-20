import argparse
import os

import yaml


def get_cfg(cfg_name="conj"):
    """
    从关键点检测的config yaml文件中读取数据转化的规则
    """
    cfg_path = "./Keypoint_Detection/nailfold_keypoint/configs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=os.path.join(cfg_path,f"{cfg_name}.yaml"), type=str, help="data file path")
    args = parser.parse_args()
    cfg_path = args.cfg

    with open(cfg_path, "r", encoding="utf8") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict