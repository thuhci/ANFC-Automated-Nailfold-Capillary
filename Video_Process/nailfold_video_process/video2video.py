import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torchvision import transforms
from Video_Process.nailfold_video_process.phase_correlation import \
    segments2trans
from Video_Process.nailfold_video_process.tools import NailfoldTools
from Video_Process.nailfold_video_process.utils.align import *
from Video_Process.nailfold_video_process.utils.align_transform import Align
from Video_Process.nailfold_video_process.utils.process_video import *
from Video_Process.nailfold_video_process.utils.torch_align import *

MAX_DISTANCE = 5
PADDING = 150
USE_KP_LOSS = 0.1125

def calculate_rough_translation(img_s, img_t, device, debug=False):
    """通过SIFT算子计算两张图片的平移

    Args:
        img_s (np.array): 源图片
        img_t (np.array): 目标图片
        device (torch.device): _description_
        debug (bool, optional): 是否开启调试输出SIFT匹配图. Defaults to False.

    Returns:
        np.array: 修正后的源图片
        float: 修正的mseloss
    """
    h, w, _ = img_s.shape
    al = Align("", "", threshold=1)
    # Extract key points and SIFT descriptors from
    # source image and target image respectively
    # img_source = (255*img_s).int().numpy().transpose(1,2,0).astype(np.uint8)
    # img_target = (255*img_t).int().numpy().transpose(1,2,0).astype(np.uint8)
    img_source = img_s.copy()
    img_target = img_t.copy()
    img_s_tensor = transforms.ToTensor()(img_s)
    img_s_tensor = img_s_tensor.to(device=device)
    img_t_tensor = transforms.ToTensor()(img_t)
    img_t_tensor = img_t_tensor.to(device=device)
    # 防止 match_SIFT不存在
    kp_s, desc_s = al.extract_SIFT(img_source)
    kp_t, desc_t = al.extract_SIFT(img_target)
    try:
        # Obtain the index of correcponding points
        fit_pos = al.match_SIFT(desc_s, desc_t)
        kp_s_ = kp_s[:, fit_pos[:, 0]]
        kp_t_ = kp_t[:, fit_pos[:, 1]] 
        dist_vec = []
        for i in range(len(fit_pos)):
            s = kp_s_[:,i]
            t = kp_t_[:,i]
            dist_vec.append(s-t)
        # 筛选所有有附近点的平移向量，非孤立的
        selected_dist_vec = []
        for i,d in enumerate(dist_vec):
            for j,v in enumerate(dist_vec):
                if abs(d[0]-v[0]) < 10 and abs(d[1]-v[1]) < 10 and i!=j:
                    selected_dist_vec.append(d)
                    break
        if len(selected_dist_vec) > 0:
            dist_vec = selected_dist_vec
            
    except:
        dist_vec = []
    
    kp_image = img_source.copy()
    for kp in kp_s_.transpose():
        kp_image = cv2.circle(kp_image, [int(p) for p in kp], 5, (255,255,255), 2)

    loss_vec = []
    # SUFT没有匹配的
    if len(dist_vec) != 0:
        # 尝试使用SUFT的关键点
        for translation in dist_vec:
            if abs(translation[0]) > w/2 - 20 or abs(translation[1]) > h/2 - 20:
                loss_vec.append(1)
                continue
            translation = np.array(translation) * np.array([2/w, 2/h])
            translation = torch.tensor(list(translation),device=device,dtype=torch.float32)
            loss = grid_affine(img_source,img_target,translation,device,img_s_tensor, img_t_tensor)
            loss_vec.append(loss)
        # 选loss最小的初值偏移量
        i = np.array(loss_vec).argmin()
        cur_lose = loss_vec[i]
        rough_translation = np.array(dist_vec[i]) * np.array([2/w, 2/h])
    else:
        cur_lose = 1
        rough_translation = None
    # 若SUFT的loss过大
    # if cur_lose > USE_KP_LOSS or rough_translation is None:
    #     print("SUFT loss:",round(cur_lose,5))
    #     _,keypoints_t = extract_keypoint(img_target,[])
    #     if debug:
    #         visualize("t-kp",img_target, [], keypoints_t)
    #     keypoints_t = [kp[0] for kp in keypoints_t]
    #     dist_vec,keypoints = extract_keypoint(img_source, keypoints_t)
    #     if debug:
    #         visualize("s-kp",img_source, [], keypoints)
    #     loss_vec = []
     
    #     # 尝试使用关键点检测
    #     for translation in dist_vec:
    #         if abs(translation[0]) > w/2 - 20 or abs(translation[1]) > h/2 - 20: 
    #             loss_vec.append(1)
    #             continue
    #         translation = np.array(translation) * np.array([2/w, 2/h])
    #         translation = torch.tensor(list(translation),device=device,dtype=torch.float32)
    #         loss = grid_affine(img_source,img_target,translation,device,img_s_tensor, img_t_tensor)
    #         loss_vec.append(loss)
    #     # 选loss最小的初值偏移量
    #     if len(loss_vec) == 0:
    #         print("Still use SUFT because of no keypoint")
    #     else:
    #         i = np.array(loss_vec).argmin()
    #         if loss_vec[i] > cur_lose:
    #             print("Still use SUFT because of higher loss")
    #         else:
    #             rough_translation = np.array(dist_vec[i]) * np.array([2/w, 2/h])
    #         print("KP loss:",round(loss_vec[i],5))
    
    return rough_translation, cur_lose, kp_image


def video2video(args, filename, rough=True, start=0, end = -1, num_iters=1, debug=False):
    GAP = 0
    imgs,fps = process_video(os.path.join(args.input, filename), args.skip_frame)
    
    end =  int(len(imgs)) + end
    print(filename, end)
    target = 0
    imgs = np.array(imgs[:end])
    
    if filename.endswith("avi"):
        output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
    else:
        output_filename = os.path.join(args.output, filename)       

    device = torch.device('cuda:2') if torch.cuda.is_available() else 'cpu'
    new_imgs = []
    record = []
    kp_images = []
    for img in tqdm.tqdm(imgs[start:end]):
        # 使用默认的迭代轮数，num_iters=1表示不进行精配准
        this_num_iters = num_iters
        translation = np.array([0,0])
        cur_lose = 1
        # 粗配准获取平移变换的初值
        if rough:
            translation, cur_lose, kp_image = calculate_rough_translation(img.copy(), imgs[target].copy(), device, debug=debug)
            kp_images.append(kp_image)
        if translation is None:
            print("Use torch affine from None")
            translation = np.array([0,0])
            this_num_iters = 800
        new_img = torch_affine(img, imgs[target], GAP=GAP, num_iters=this_num_iters, translation=translation, device=device, debug=debug)
        new_imgs.append(new_img)
        record.append(cur_lose)
    os.makedirs(args.output,exist_ok=True)
    imgs2video(new_imgs, output_filename, int(fps / args.skip_frame))
    # if debug:
    imgs2video(kp_images, output_filename.replace(".mp4","-kp.mp4"), int(fps / args.skip_frame))
    

def visualize_keypoints(img, keypoints):
    """将关键点可视化

    Args:
        img (_type_): _description_
        keypoints (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = img.copy()
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image, tuple(kp), 1, (255,255,255), 1)
    return image


def keypoints2trans(keypoints_s, keypoints_t, method="permute"):
    """将两组关键点列表，转化为可信的平移量

    Args:
        keypoints_s (_type_): _description_
        keypoints_t (_type_): _description_

    Returns:
        _type_: dist_vec, selected_dist_vec
    """
    # 使用穷举匹配
    if method == 'permute':
        dist_vec = []
        for target in keypoints_t:
            top_target = target[0]
            for kp in keypoints_s:
                top = kp[0]
                translation = np.array(top_target) - np.array(top)
                dist_vec.append(translation)
        # 筛选所有有附近点的平移向量，非孤立的
        selected_dist_vec = []
        # dist_vec = np.array(dist_vec)
        # plt.scatter(dist_vec[:,0],dist_vec[:,1])
        # plt.savefig("a.png")
        for i,d in enumerate(dist_vec):
            for j,v in enumerate(dist_vec):
                if abs(d[0]-v[0]) < 8 and abs(d[1]-v[1]) < 8 and i!=j and abs(d[0]-v[0])+abs(d[1]-v[1]) < 10:
                    selected_dist_vec.append(d)
                    break
        near_dist_cnt = []
        if len(selected_dist_vec)>0:
            # 检验筛选出的关键点平移量之间是否相近
            selected_dist_vec = np.array(selected_dist_vec)
            for i,d in enumerate(selected_dist_vec):
                cnt = 0
                for j,v in enumerate(selected_dist_vec):
                    if abs(d[0]-v[0]) > 8 or abs(d[1]-v[1]) > 8 or abs(d[0]-v[0])+abs(d[1]-v[1]) > 10:
                        cnt += 1
                near_dist_cnt.append(cnt)
            near_dist_cnt = np.array(near_dist_cnt)
            # 选取和其他可信偏移量距离远的最少的那些偏移量
            selected_dist_vec = selected_dist_vec[near_dist_cnt==near_dist_cnt.min()]
        return dist_vec, selected_dist_vec
    # 使用ransac随机匹配
    elif method=='ransac':
        from Video_Process.nailfold_video_process.utils.affine_ransac import \
            Ransac
        ransac = Ransac(K = 4, threshold= 50)
        A, t, inliers = ransac.ransac_fit(keypoints_s[:,0,...], keypoints_t[:,0,...])
        return [t], [t]


def video2video_by_detect(args, filename, start=0, end = -1, debug=False):
    """使用关键点检测的关键点进行配准

    Args:
        args (_type_): _description_
        filename (_type_): 文件名
        start (int, optional): 开始. Defaults to 0.
        end (int, optional): 结束. Defaults to -1.
        debug (bool, optional): _description_. Defaults to False.
    """
    RANGE = 10
    imgs,fps = process_video(os.path.join(args.input, filename), args.skip_frame)
    tool = NailfoldTools()
    end =  int(len(imgs)) + end
    print(filename, end)
    target = 0
    imgs = np.array(imgs[start:end])
    
    if filename.endswith("avi"):
        output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
    else:
        output_filename = os.path.join(args.output, filename)
    
    # 查询储存的关键点文件
    kp_file = output_filename.replace(".mp4","-kp.json")
    if os.path.exists(kp_file):
        with open(kp_file,"r") as f:
            kp_json = json.load(f)
            all_keypoints = [np.array(kp) for kp in kp_json['kp']]
    # 没有计算过关键点，则计算关键点并储存
    else:      
        # 只截取中央部分提取关键点，减少关键点数量
        all_keypoints, all_boxes = tool.imgs2keypoint(imgs)
        with open(kp_file,"w") as f:
            json.dump({"kp":[kp.tolist() for kp in all_keypoints]},f)
    # 将关键点可视化在原照片上
    kp_imgs = [visualize_keypoints(img, all_keypoints[i]) for i,img in enumerate(imgs)]
    imgs2video(kp_imgs, output_filename.replace(".mp4", "-kp.mp4"), int(fps / args.skip_frame))
    all_trans = []
    new_imgs = []
    it = tqdm.tqdm(range(len(all_keypoints)))
    it.set_description("select vector")
    # 先计算和target图片的关键点之间的平移量是否存在同一性
    for i in it:
        dist_vec, selected_dist_vec = keypoints2trans(all_keypoints[i], all_keypoints[target])
        if len(selected_dist_vec) > 0:
            translation = np.array(selected_dist_vec).mean(axis=0)
            translation = np.expand_dims(translation,0)
            translation = translation.transpose()
            all_trans.append(translation)
            new_img = trans_affine(imgs[i], translation)
            new_imgs.append(new_img)
        else:
            all_trans.append(None)
            new_imgs.append(None)
    # 若没能计算偏移量，则和附近10张图片关键点计算偏移量
    for i in range(len(all_keypoints)):
        if all_trans[i] is None:
            for j in range(max(0,i-RANGE),min(i+RANGE, len(all_keypoints))):
                if all_trans[j] is None:
                    continue
                dist_vec, selected_dist_vec = keypoints2trans(all_keypoints[i], all_keypoints[j])
                if len(selected_dist_vec) > 0:
                    translation = np.array(selected_dist_vec).mean(axis=0)
                    translation = np.expand_dims(translation,0)
                    translation = translation.transpose()
                    translation += all_trans[j]
                    all_trans[i] = translation
                    new_img = trans_affine(imgs[i], translation)
                    new_imgs[i] = new_img
                    break
    # 若实在没有偏移量，则通过附近对称的两个偏移量取平均推测
    for i in range(len(all_keypoints)):
        if all_trans[i] is None:
            translation = None
            j=0
            while j <= i:
                j+=1
                if i+j == len(all_trans):
                    break
                if all_trans[i-j] is not None and all_trans[i+j] is not None:
                    translation = (all_trans[i-j] + all_trans[i+j])/2
                    break
            if translation is None:
                new_imgs[i] = np.zeros_like(imgs[i-1])
                new_imgs[i][...,:] = imgs[i-1].mean(axis=0).mean(axis=0).astype(np.uint8)
            else:
                all_trans[i] = translation
                new_img = trans_affine(imgs[i], translation)
                new_imgs[i] = new_img
            # new_imgs[i] = np.zeros_like(imgs[i-1])
            # new_imgs[i][...,:] = imgs[i-1].mean(axis=0).mean(axis=0).astype(np.uint8)
    
    imgs2video(new_imgs, output_filename, int(fps / args.skip_frame))


def avi2mp4(args, filename):
    """
    将avi转化为mp4格式
    """
    imgs,fps = process_video(os.path.join(args.input, filename), args.skip_frame)
    if len(imgs) == 0:
        return
    if filename.endswith("avi"):
        output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
    else:
        output_filename = os.path.join(args.output, filename)
    imgs2video(imgs, output_filename, int(fps / args.skip_frame))


def video2video_by_detect_and_phase(args, filename, start=0, end = -1, debug=False):
    """使用关键点检测的关键点、相位相关，进行配准

    Args:
        args (_type_): _description_
        filename (_type_): 文件名
        start (int, optional): 开始. Defaults to 0.
        end (int, optional): 结束. Defaults to -1.
        debug (bool, optional): _description_. Defaults to False.
    """
    RANGE = 10
    imgs,fps = process_video(os.path.join(args.input, filename), args.skip_frame)
    tool = NailfoldTools()
    end =  int(len(imgs)) + end
    print(filename, end)
    target = 0
    imgs = np.array(imgs[start:end])
    
    if filename.endswith("avi"):
        output_filename = os.path.join(args.output, filename.replace('avi','mp4'))
    else:
        output_filename = os.path.join(args.output, filename)
    
    # 查询储存的关键点文件
    kp_file = output_filename.replace(".mp4","-kp.json")
    if os.path.exists(kp_file):
        with open(kp_file,"r") as f:
            kp_json = json.load(f)
            all_keypoints = [np.array(kp) for kp in kp_json['kp']]
    # 没有计算过关键点，则计算关键点并储存
    else:      
        # 只截取中央部分提取关键点，减少关键点数量
        all_keypoints, all_boxes = tool.imgs2keypoint(imgs)
        with open(kp_file,"w") as f:
            json.dump({"kp":[kp.tolist() for kp in all_keypoints]},f)
    # 将关键点可视化在原照片上
    kp_imgs = [visualize_keypoints(img, all_keypoints[i]) for i,img in enumerate(imgs)]
    imgs2video(kp_imgs, output_filename.replace(".mp4", "-kp.mp4"), int(fps / args.skip_frame))
    all_trans = []
    it = tqdm.tqdm(range(len(all_keypoints)))
    it.set_description("select vector")
    # 先计算和target图片的关键点之间的平移量是否存在同一性
    for i in it:
        dist_vec, selected_dist_vec = keypoints2trans(all_keypoints[i], all_keypoints[target])
        if len(selected_dist_vec) > 0:
            translation = np.array(selected_dist_vec).mean(axis=0)
            translation = np.expand_dims(translation,0)
            translation = translation.transpose()
            all_trans.append(translation)
        else:
            all_trans.append(None)

    imgs = np.array(imgs)
    detect_method_ready = np.zeros(len(all_trans), np.bool8)
    trans_list = np.zeros([len(all_trans)] + list(all_trans[0].shape))
    for i,t in enumerate(all_trans):
        if (t is None): 
            detect_method_ready[i] = 1
        else:
            trans_list[i] = t
    
    # 对于关键点无法匹配的图片进行相位相关
    phase_imgs =np.concatenate([ np.array([imgs[0]]), imgs[detect_method_ready] ], axis=0)
    if np.any(detect_method_ready):
        segments = tool.imgs2segments(phase_imgs)
        phase_trans_list, scores = segments2trans(segments, 0.2, output_filename=output_filename)
        trans_list[detect_method_ready] = phase_trans_list[1:]

    new_imgs = imgs2imgs_by_trans(imgs, trans_list)
    imgs2video(new_imgs, output_filename, int(fps / args.skip_frame))


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    args.output = "/home/user/nailfold/Tangshan-Samples/videos"#"./"
    args.input = "videos/7.28/55896"
    filename = "wmv1.mp4"
    video2video_by_detect_and_phase(args,filename)
    # for filename in os.listdir(args.input):
    #     if not (filename.endswith("mp4") or filename.endswith("avi")):
    #         continue
    #     video2video_by_detect(args,filename)
        