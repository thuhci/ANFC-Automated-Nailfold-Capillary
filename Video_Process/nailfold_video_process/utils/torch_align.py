from torch.nn import functional as F
from sklearn import metrics
import torch
import numpy as np
from torchvision import transforms
import tqdm
from Video_Process.nailfold_video_process.utils.process_video import *

def mutual_information_loss(img_s,img_t):
    if type(img_s) is torch.Tensor:
        return 1/metrics.mutual_info_score(img_s.cpu().numpy().flatten(),img_t.cpu().numpy().flatten())
    return 1/metrics.mutual_info_score(img_s.flatten(),img_t.flatten())

def correlation_coefficient(img_s,img_t):
    if type(img_s) is torch.Tensor:
        return 1/np.corrcoef(img_s.cpu().numpy().flatten(),img_t.cpu().numpy().flatten())[0][1]
    return 1/np.corrcoef(img_s, img_t)[0][1]

def grid_affine(img_s, img_t, translation, device, img_s_tensor=None, img_t_tensor=None):
    """pytorch图像配准的一个iteration

    Args:
        img_s (np.array): _description_
        img_t (np.array): _description_
        translation (torch.Tensor): _description_
        device (_type_): _description_

    Returns:
        torch.Tensor: loss
    """
    GAP = 0
    if img_s_tensor is None:
        img_s = transforms.ToTensor()(img_s)
        img_s = img_s.to(device=device)
    else:
        img_s = img_s_tensor
    if img_t_tensor is None:
        img_t = transforms.ToTensor()(img_t)
        img_t = img_t.to(device=device)
    else:
        img_t = img_t_tensor
    _, h, w = img_s.size()
    loss_func = torch.nn.MSELoss()
    theta = torch.eye(2,device=device)
    theta = torch.cat([theta,translation.unsqueeze(0).view(2,1)], dim=-1)
    theta = theta.to(device=device)
    grid = F.affine_grid(theta.unsqueeze(0), img_s.unsqueeze(0).size())
    output = F.grid_sample(img_s.unsqueeze(0), grid)
    
    trans = torch.Tensor([w/2, h/2]).to(device=device) * translation.detach()
    range_h = [max(-int(trans[1]), 0), h - max(int(trans[1]), 0)]
    range_w = [max(-int(trans[0]), 0), w - max(int(trans[0]), 0)]
    # range_h = [max(abs(int(trans[1])) + 50,GAP), h - max(abs(int(trans[1])) + 50,GAP)]
    # range_w = [max(abs(int(trans[0])) + 50,GAP), w - max(abs(int(trans[0])) + 50,GAP)]
    if range_h[0] > range_h[1] or range_w[0] > range_w[1]:
        print(range_w, range_h)
        return 1
    new_img_s = output[0]
    new_img_s = new_img_s[...,range_h[0]:range_h[1],range_w[0]:range_w[1]]
    img_tar = img_t[...,range_h[0]:range_h[1],range_w[0]:range_w[1]]
    loss = loss_func(new_img_s, img_tar)
    return loss.item()



# to refine a image img_s,img_t
# 精配准受到图像中高亮斑移动的影响极大
def torch_affine(img_s, img_t, GAP=75, translation=np.array([0,0]), device="cpu", num_iters=400, debug=False):
    """使用torch自带的affine sample进行梯度下降的图像配准

    Args:
        img_s (np.array): 待配准对象
        img_t (np.array): 目标图片
        translation (np.array): 初始平移
        GAP (int, optional): 截取掉图片四周的宽度. Defaults to 75.
        debug (bool, optional): 是否输出测试视频. Defaults to False.

    Returns:
        np.array: 配准后图片
    """
    PADDING = 50
    img_t_np = img_t.copy()
    img_s = transforms.ToTensor()(img_s)
    img_t = transforms.ToTensor()(img_t)
    _, h, w = img_s.size()
    loss_func = torch.nn.MSELoss()
    
    img_torch = img_s.to(device=device)
    
    translation = torch.nn.Parameter(torch.tensor(translation.tolist(),device=device,dtype=torch.float32))
    # 进行pytorch的精配准，lr太大会大大偏出, 0.005在纯affine时表现良好
    optimizer = torch.optim.Adam([translation],lr=0.001)
    # num_iters = 200
    it = range(num_iters)
    if debug:
        it = tqdm.tqdm(it)
    img_t = img_t.to(device)
    if debug:
        debug_imgs = [img_t_np]
    for i in it:
        theta = torch.eye(2,device=device)
        theta = torch.cat([theta,translation.unsqueeze(0).view(2,1)], dim=-1)
        theta = theta.to(device=device)
        grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
        output = F.grid_sample(img_torch.unsqueeze(0), grid)
        
        trans = torch.Tensor([w/2, h/2]).to(device=device) * translation.detach()
        range_h = [max(-int(trans[1]), 0), h - max(int(trans[1]), 0)]
        if trans[1] < 0:
            range_h = [max(range_h[0], PADDING), range_h[1]]
        else:
            range_h = [range_h[0], min(range_h[1], h-PADDING)]
        range_w = [max(-int(trans[0]), 0), w - max(int(trans[0]), 0)]
        if trans[0] < 0:
            range_w = [max(range_w[0], PADDING), range_w[1]]
        else:
            range_w = [range_w[0], min(range_w[1], w-PADDING)]
        # range_h = [max(abs(int(trans[1])) + PADDING,GAP), h - max(abs(int(trans[1])) + PADDING,GAP)]
        # range_w = [max(abs(int(trans[0])) + PADDING,GAP), w - max(abs(int(trans[0])) + PADDING,GAP)]
        new_img_torch = output[0]
        new_img_torch = new_img_torch[...,range_h[0]:range_h[1],range_w[0]:range_w[1]]
        img_tar = img_t[...,range_h[0]:range_h[1],range_w[0]:range_w[1]]
        loss = loss_func(new_img_torch, img_tar)
        if i==0:
            first_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        # it.set_description(f"loss: {loss.item()}")
        # image compare loss is too much, cant be optimized
        if loss.isnan():
            break
        if debug and (i < 200 or i == len(it)-1):
            it.set_description(f"loss: {loss.item()}")
            debug_output = output[0].cpu()[...,GAP:h-GAP,GAP:w-GAP]
            debug_imgs.append(255*debug_output.detach().numpy().transpose(1,2,0))  
        optimizer.step()
    print("before: %.5f" % first_loss, "after: %.5f" % loss.item())
    if debug and first_loss > 0.001:
        debug_imgs.append(img_t_np)
        print("output debug video...")
        imgs2video(debug_imgs, "debug.mp4", 100)
    new_img_torch = output[0].cpu()[...,GAP:h-GAP,GAP:w-GAP]
    # cv2.imwrite(f'./aligned/{num}.png',255*new_img_torch.detach().numpy().transpose(1,2,0)[...,::-1])
    return 255*new_img_torch.detach().numpy().transpose(1,2,0)  
