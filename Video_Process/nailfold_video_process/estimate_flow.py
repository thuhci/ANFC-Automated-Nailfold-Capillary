import json
import os

import cv2
import numpy as np
# from Image_Analysis.nailfold_image_profile.image_analysis import \
#     kpimg_morphology_parameters
from Video_Process.nailfold_video_process.tools import NailfoldTools
from Video_Process.nailfold_video_process.utils.distance_geometry import *
from Video_Process.nailfold_video_process.utils.flow_detection import *
from Video_Process.nailfold_video_process.utils.morph import *
from Video_Process.nailfold_video_process.utils.process_video import *
from Video_Process.nailfold_video_process.utils.reconnect import *


def draw_bbox_rectangle(newimg, bbox):
    return cv2.rectangle(newimg.copy(), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (bbox[0],bbox[1]), (255, 0, 0), 1, 4)

def draw_line(img, start, end):
    return cv2.line(img.copy(),[int(p) for p in start],[int(p) for p in end],(255,255,255),1)

def draw_arrow(img, point, angle):
    LENGTH = 10
    angle = angle * np.pi / 180
    end = (point[0] + int(np.cos(angle) * LENGTH), point[1]+int(np.sin(angle) * LENGTH))
    return cv2.arrowedLine(img.copy(), point, end, (0,0,255),1,8,0,0.3)


def visualize_angle_map(name, thinned, angle_map, type=1):
    if type == 0:
        blank = np.zeros_like(angle_map).astype(np.uint8)
        blank = cv2.cvtColor(blank,cv2.COLOR_GRAY2BGR)
        h,w = angle_map.shape
        thinned = cv2.dilate(thinned, np.ones((5,5),np.uint8))
        BLOCK = 5
        for x in range(h):
            for y in range(w):
                if angle_map[x][y] == -10:
                    blank = cv2.circle(blank, (y,x), 1, (0,255,0), 1)
                    continue
                if x% BLOCK!=0 or y%BLOCK!=0:
                    continue
                if thinned[x][y] == 0:
                    continue
                angle = lookup_angle((x,y),angle_map)
                if angle > 0:
                    blank = draw_arrow(blank, (y,x), angle)
        cv2.imwrite(name, blank)
    else:
        blank = np.zeros_like(angle_map).astype(np.uint8)
        slope =  255 / angle_map.max()
        blank[angle_map >= 0] = (angle_map[angle_map >= 0] * slope).astype(np.uint8)
        blank = cv2.cvtColor(blank,cv2.COLOR_GRAY2BGR)
        blank[angle_map == -10] = [0,255,0]
        cv2.imwrite(name, blank)

def get_visualize_dis_map(dis_map, binarysegment):
    visualize_dis_map = (dis_map-(np.min(dis_map))).astype(int)*255/(np.max(dis_map)-np.min(dis_map))
    visualize_dis_map = cv2.cvtColor(visualize_dis_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    visualize_dis_map[binarysegment,0] = 0
    visualize_dis_map[binarysegment,1] = 10
    visualize_dis_map[~binarysegment] = 0
    return visualize_dis_map


def get_angle_map(ori_segment, thinned, debug=False):
    """计算切向场、并根据切向场重新链接、再次计算切向场

    Args:
        ori_segment (_type_): 分割灰度图片
        thinned (_type_): 骨架线灰度图片
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: thinned, angle_map
    """
    # 计算血管的切向场
    angle_map = contours2angle(ori_segment, thinned)
    if debug:
        visualize_angle_map("angle.png",thinned,angle_map,1)
    if debug:
        cv2.imwrite("thinned.png", thinned)
    # 将断裂处重新链接
    thinned = reconnect_skeleton(thinned, angle_map)
    if debug:
        cv2.imwrite("reconnect-thinned.png", thinned) 
    # 再次计算切向场   
    angle_map = contours2angle(ori_segment, thinned)
    if debug:
         visualize_angle_map("reconnect-angle.png",thinned,angle_map,1)
    return thinned, angle_map


def estimate_flow(args, i, white_threshold=[15], debug=False, visualize=True):
    """识别白细胞并计算血管内流速

    Args:
        args (_type_): _description_
        i (int): 计算kp-i号视频
        white_threshold (int): 白细胞的识别阈值--绿色值比上一帧大15
    """
    with open(os.path.join(args.input,"kps.json"), "r") as file:
        json_kps = json.load(file)
    json_kp = json_kps['kp-videos'][i]
    filename = json_kp['path']
    kps = json_kp['kp']
    os.makedirs(args.output,exist_ok=True)
    # filename = "./Video_Process/data/aligned_videos/2101011647.mp4"
    
    imgs,fps = process_video(filename, args.skip_frame)

    new_imgs = []

    sample = []    
    
    # 相邻两帧相减
    imgs = np.array(imgs).astype(int)
    feature = imgs[1:] - imgs[:-1]
    # 获取血管部分 G比R大50
    # segment = imgs[1:][..., 2] - imgs[1:][..., 1] > 50
    # segment = segment.astype(np.uint8)
    erode_kernel = np.ones((3,3), np.uint8)
    dilate_kernel = np.ones((3,3),np.uint8)
    # 获取血管部分 U-net
    if os.path.exists(os.path.join(args.input,f"ori-segment-{i}.png")):
        ori_segment = cv2.imread(os.path.join(args.input,f"ori-segment-{i}.png"))
        # 计算中心线
        ori_segment = cv2.cvtColor(ori_segment, cv2.COLOR_BGR2GRAY)
        contours, thinned = skeleton_by_thin(ori_segment.copy())
        # segment = cv2.imread(os.path.join(args.input,f"segment-{i}.png"))
        # thinned = segment.copy()
        # segment = cv2.cvtColor(segment,cv2.COLOR_BGR2GRAY)
    else:
        tool = NailfoldTools()
        ori_segment = tool.kpimage2segment(imgs[1])
        
        # ori_segment = cv2.erode(ori_segment,erode_kernel,iterations = 1)
        cv2.imwrite(os.path.join(args.input,f"ori-segment-{i}.png"), ori_segment)
        # thinned = cv2.cvtColor(ori_segment,cv2.COLOR_GRAY2RGB)
        # 计算中心线并膨胀
        contours, thinned = skeleton_by_thin(ori_segment.copy())
        
        dilate_kernel = np.ones((3,3),np.uint8)
        segment = cv2.dilate(thinned,dilate_kernel,iterations = 1)
        cv2.imwrite(os.path.join(args.input,f"segment-{i}.png"), segment)
    # segment = np.unsqueeze()

    
    # 计算血管的切向场
    thinned, angle_map = get_angle_map(ori_segment, thinned, debug)
    # 同时，检查骨架线是否有错误的链接
    angle_map, thinned = refine_angle_map(angle_map,thinned, ori_segment, debug=debug)
    if debug:
        cv2.imwrite("refine-thinned.png", thinned) 
    if debug:
        visualize_angle_map("refine-angle.png",thinned,angle_map,1)

    angle_map = remove_cross_zone(angle_map,debug=debug)
    if debug:
        visualize_angle_map("remove-angle.png",thinned,angle_map,1)
    
    dilate_kernel = np.ones((3,3),np.uint8)
    segment = cv2.dilate(thinned,dilate_kernel,iterations = 1)
    
    binarysegment = segment.astype(np.bool8)
    # 计算血管的距离场
    dis_map = smart_get_dis_map(x = kps[0], y = kps[1], binarysegment=binarysegment, angle_field=angle_map, 
                                visualize=get_visualize_dis_map, debug=debug)
    visualize_dis_map = get_visualize_dis_map(dis_map, binarysegment)
    # 处理其他血管的mask
    binarysegment[dis_map==-1] = False
    if debug:
        cv2.imwrite("distance.png", visualize_dis_map)
    base_imgs = []
    imgs = imgs[1:]
    for id_s,img in enumerate(imgs):
        img = img.astype(np.uint8)
        compare = cv2.add(img, visualize_dis_map)
        for kp in kps:
            compare = cv2.circle(compare.copy(), [int(p) for p in kp], 2, (255,0,0), 1)
        base_imgs.append(np.concatenate([img,compare],axis=1))
    
    threshold_imgs = []
    white_threshold = np.array(white_threshold)
    white_threshold.sort()
    white_threshold = white_threshold[::-1]
    for w_t in white_threshold:
        # 白细胞特征：绿色比上一帧大5
        binanrymask = (feature[...,1] > w_t) # | (feature[...,1] < -5)
        mask = binanrymask & binarysegment
        mask = mask.astype(np.uint8)
        if w_t < 20:
            dilate_kernel = np.ones((5,5),np.uint8)
            mask = [cv2.erode(m,erode_kernel,iterations = 1) for m in mask]
            mask = [cv2.dilate(m,dilate_kernel,iterations = 1) for m in mask]
        
        mask = np.array(mask)
        color_mask = np.expand_dims(mask,axis=-1)
        color_mask = np.broadcast_to(color_mask,list(color_mask.shape[:-1])+[3])
        color = np.zeros_like(color_mask)
        color[...,1] = color_mask[...,1]*255
        color= color.astype(np.uint8)
        
        
        # 中心线
        # lines = line_detect(thinned)
        
        framesbboxs = []
        new_imgs = []
        for id_s,img in enumerate(imgs):
            img = img.astype(np.uint8)
            newimg = cv2.add(img,color[id_s])
            # 获取该帧的白细胞分割对应所有包围盒(w,h,dw,dh)
            bboxs = mask2bbox(mask[id_s])
            for id_b,j in enumerate(bboxs):
                newimg = draw_bbox_rectangle(newimg.copy(), j)
            new_imgs.append(newimg)
            framesbboxs.append(bboxs)
        
        # 每帧的所有白细胞首尾距离值
        cell_range = [[check_bbox_distance(bbox,dis_map) for bbox in bboxs] for bboxs in framesbboxs]
        # 每帧的所有白细胞点坐标(w,h)
        white_kps = [[bbox2kp(bbox) for bbox in bboxs] for bboxs in framesbboxs]
        nv = NailfoldVideo(white_kps, dis_map, cell_range, white_threshold=w_t)
        # all_path = nv.find_all_path(depth=4)
        all_flow = nv.find_all_flow(depth=4)
        all_flow = nv.unify_all_flow(all_flow)

        if visualize:
            graph = nv.visualize(all_flow)
            # imgs2video(graph, os.path.join(args.output, f"graph-kp-{i}.mp4"), 20)

            # for path in all_path:
            #     # path (distance, frame, id) white_kps
            #     last = None
            #     for distance, frame, id in path:
            #         now = white_kps[frame][id]
            #         if last is not None:
            #             new_imgs[frame] = draw_line(new_imgs[frame], last, now)
            #         last = now
                
                
            new_imgs = np.array(new_imgs)       
            graph = np.array(graph)
            blank = np.zeros([imgs.shape[0], 300, 200, 3], np.uint8)
            blank[:, :new_imgs.shape[1], :new_imgs.shape[2]] = new_imgs
            new_imgs = np.concatenate([blank, graph],axis=-2)
            threshold_imgs.append(new_imgs)
            
        if len(all_flow) > 2:
            break  
    if visualize:
        base_imgs = np.array(base_imgs)
        blank = np.zeros([base_imgs.shape[0], 300, base_imgs.shape[2], 3], np.uint8)
        blank[:, :base_imgs.shape[1]] = base_imgs
        new_imgs = np.concatenate([blank]+threshold_imgs,axis=-2)
        imgs2video(new_imgs, os.path.join(args.output, f"flow-kp-{i}.mp4"), 20)

    data = dict()
    # 血管流动检测结果
    if len(all_flow) > 2:
        data.update(flow2data(all_flow))
    # 血管精细形态指标检测结果
    # data.update(kpimg_morphology_parameters(np.array([kps]), dis_map))
    return data


def flow2data(all_flow):
    """通过流动检测结果，输出该血管的数据

    Args:
        all_flow (list(WhiteCellFLow)): 本视频的所有流动Flow

    Returns:
        data: {
            'velocity'
        }
    """
    speed = np.array([flow.get_mean() for flow in all_flow])
    speed = np.abs(speed)

    white_cell_count = len(all_flow)
    data = {
        'velocity': [speed.min(), speed.mean(), speed.max()],
        'white_cell': white_cell_count
    }
    return data


if __name__ == '__main__':
    args = parse_args()
    args.output = args.input = "debug/pipeline/cache/wmv1-aligned"
    i = 2
    estimate_flow(args, i, white_threshold=[5], debug=False)