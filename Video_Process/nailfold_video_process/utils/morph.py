# 形态学算法库
# 包括 求方向场、求骨架线、mask求包围盒
import numpy as np
import cv2

CROSS_SIGN = -10

def contours2angle(img, blank):
    """通过骨架线求出血管方向场
    1. 若该位置没有骨架，则方向角度为-1
    2. 若该位置是血管交叉点，则方向角度为-10
    3. 目前若是水平方向，角度难以估计正确，因为0-180角度首尾不连续

    Args:
        img (np.array): 源图片,灰度
        contours (list): 骨架线

    Returns:
        angle_field(np.array): 每一点的方向角度 0-180
    """
    blank = blank.astype(np.bool8)
    h,w = blank.shape
    # 属于0-180
    angle_field = np.zeros_like(blank, np.int64)
    PADDING = 5
    MARGIN = 3
    INNER = 2

    y_field = np.broadcast_to(np.arange(-PADDING, PADDING).reshape((2*PADDING,1)), (2*PADDING,2*PADDING))
    x_field = np.broadcast_to(np.arange(-PADDING, PADDING), (2*PADDING,2*PADDING))
    angle = np.arctan( y_field/(x_field+1e-8) ) * (180 / np.pi)
    angle[angle < 0] = angle[angle < 0] + 180

    for y in range(h):
        for x in range(w):
            if not blank[y][x]:
                angle_field[y][x] = -1
                continue
            if y+PADDING >= h or y-PADDING <= 0 or x+PADDING >= w or x-PADDING <= 0:
                angle_field[y][x] = -1
                continue 
            mask = blank[y-PADDING:y+PADDING, x-PADDING:x+PADDING].copy()
            ori_mask = mask.copy()
            mask[PADDING-INNER:PADDING+INNER,PADDING-INNER:PADDING+INNER] = False
            # 构造一个边框
            margin = np.zeros_like(mask,np.bool8)
            margin[PADDING-MARGIN-1:PADDING+MARGIN+1, PADDING-MARGIN-1:PADDING+MARGIN+1] = True
            margin[PADDING-MARGIN:PADDING+MARGIN, PADDING-MARGIN:PADDING+MARGIN] = False
            
            angles = angle[mask].flatten()
            angles = np.sort(angles)
            if angles.shape[0] < 2:
                continue
            mean = angles.mean()
            
            margin = margin & ori_mask
            try:
                contours, hierarchy = cv2.findContours(margin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours, hierarchy = cv2.findContours(margin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            deviate = len(contours)

            if deviate > 2:
                angle_field[y][x] = -10
            else:
                angle_field[y][x] = mean
    cross_zone = (angle_field==-10)
    cross_zone = cv2.dilate(cross_zone.astype(np.uint8), np.ones((3,3), np.uint8)).astype(np.bool8)
    angle_field[(angle_field >= 0) & cross_zone] = -10
    return angle_field


def cross_zone2cross_point(thinned, cross_point):
    """从交叉点的segmentation获取精确的交叉点坐标列表

    Args:
        thinned (_type_): _description_
        cross_point (_type_): _description_
    
    Returns:
        cross_point_list-(x,y): thinned[x][y]
    """
    
    margin = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ],np.bool8)
    
    loop = [
        [0,0],
        [0,1],
        [0,2],
        [1,2],
        [2,2],
        [2,1],
        [2,0],
        [1,0],
        [0,0],
    ]
    
    point_list = np.where(cross_point)
    point_list = np.array(point_list).transpose()
    cross_point_list = []
    for x,y in point_list:
        ori_mask = thinned[x-1:x+2,y-1:y+2].astype(np.bool8)
        new_margin = margin & ori_mask
        last = None
        cnt = 0
        for u,v in loop:
            if last is not None:
                if last != new_margin[u][v]:
                    cnt += 1
            last = new_margin[u][v]
        if cnt == 6:
            cross_point_list.append((x,y))
    return cross_point_list


def skeleton_by_thin(image, output_name=None):
    """获取mask的骨架

    Args:
        image (灰度): _description_

    Returns:
        _type_: contours, thinned
    """
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thinned = cv2.ximgproc.thinning(binary)
    try:
        contours, hierarchy = cv2.findContours(thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, contours, hierarchy = cv2.findContours(thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 1, 8)
    if output_name is not None:
        cv2.imwrite("skeleton.png", image)
    return contours, thinned


def line_detect(image):
    """检测图像中的直线并求出角度

    Args:
        image (np.array): 彩色图像

    Returns:
        lines: 直线的列表 (angle, line(x1,y1,x2,y2))
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 40,minLineLength=1,maxLineGap=10)
    # 对通过霍夫变换得到的数据进行遍历
    result = []
    for line in lines:
        # newlines1 = lines[:, 0, :]
        x1,y1,x2,y2 = line[0]  #两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
        cv2.line(image,(x1,y1),(x2,y2),(255,255,255),1)   #在原图上画线
        # 转换为浮点数，计算斜率
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)
        print("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
        if x2 - x1 == 0:
            result.append((90, line))
        elif y2 - y1 == 0 :
            result.append((0, line))
        else:
            # 计算斜率
            k = -(y2 - y1) / (x2 - x1)
            # 求反正切，再将得到的弧度转换为度
            result.append((np.arctan(k) * 57.29577, line)) 
    #   显示最后的成果图
    cv2.imwrite("line_detect.png",image)
    return result


def mask2bbox(mask):
    """将分割结果求出其包围盒

    Args:
        mask (np.array): 灰度图像

    Returns:
        bboxes: 包围盒的列表
    """
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]