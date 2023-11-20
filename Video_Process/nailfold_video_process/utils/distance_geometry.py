# 生成血管状mask的距离场
import cv2
import math
import numpy as np

from Video_Process.nailfold_video_process.utils.morph import cross_zone2cross_point

CROSS_SIGN = -10
FIRST_CROSS_SIGN = 200

LINE_ZONE = 0
LOOP_ZONE = 1
CROSS_ZONE = 100

# 单连通的cross点之间的最大距离
MAX_LINE_DISTANCE_CROSS = 20
# 环的cross点之间的最大距离
MAX_LOOP_DISTANCE_CROSS = 10
# 不搜索靠近
BFS_PADDING = 20

def bbox2kp(bbox):
    return bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2


def get_distance(x,y):
    return math.sqrt(pow(x[0]-y[0],2)+pow(x[1]-y[1],2))


def check_bbox_distance(bbox, dis_map):
    """
    查询一个包围盒内，距离场的最大值和最小值
    returns: [min, max]
    """
    zone = dis_map[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    return [(zone[zone > 0]).min(), (zone[zone > 0]).max()]


def visualize_sign_map(sign_map, graph):
    visual = np.zeros_like(sign_map,np.uint8)
    visual = cv2.cvtColor(visual,cv2.COLOR_GRAY2BGR)
    visual[sign_map==CROSS_ZONE] = [255,0,0]
    visual[sign_map==LINE_ZONE] = [0,255,0]
    visual[sign_map==LOOP_ZONE] = [0,0,255]
    visual[graph==FIRST_CROSS_SIGN] = [255,255,255]
    visual[sign_map==-1] = [255,255,0]
    return visual


def refine_angle_map(angle_map, thinned, ori_segment, debug=False):
    """将角度场进行优化，同时去除骨架线的误连接：
    将两个交叉点之间的较短单连通区域、或极短的双连通区域的一条边
    统一视作同一个交叉点

    Args:
        angle_map (np): 角度场
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        angle_map: 角度场
    """
    new_angle_map = angle_map.copy()
    cross = (angle_map == CROSS_SIGN)
    label = np.zeros_like(angle_map,np.uint8)
    h,w = label.shape
    # 获取交叉的中心点
    cross_point_list = cross_zone2cross_point(thinned, cross)
    # label标记是否访问，255为不需要访问
    for x,y in cross_point_list:
        # graph为BFS_path搜索的图
        graph = angle_map.copy()
        if label[x][y] == 255:
            continue
        # label[x][y] = 0
        # 将该点周围的同一cross区域均置label为255、置graph为FIRST_CROSS_SIGN
        local_cross = BFS(graph==CROSS_SIGN, (x,y), (0,0,0,0,0), direction=True) >= 0
        graph[local_cross] = FIRST_CROSS_SIGN
        label[local_cross] = 255
        
        sign_map, from_map, dis_map = BFS_path(graph, (x,y))
        
        if debug:
            visual = visualize_sign_map(sign_map, graph)
            cv2.imwrite(f"c({x},{y}).png",visual)
        # 寻找单连通的cross区域的中心点
        single_cross = []
        for x1,y1 in cross_point_list:
            if sign_map[x1][y1] == CROSS_ZONE:
                single_cross.append((x1,y1))
        # 从CROSS区域交叉点开始回溯，直到。。。
        for x2,y2 in single_cross:
            u,v = x2,y2
            # 是否找到路径
            is_path = False
            cross_path = np.zeros_like(angle_map, np.bool8)
            while 1:
                new_angle_map[u][v] = CROSS_SIGN
                cross_path[u][v] = True
                # 回溯到起始点
                if from_map[u][v] is None:
                    is_path = True
                    break
                u,v = from_map[u][v]
                
                # 只回溯到 单连通的、有角度的
                if sign_map[u][v] == LINE_ZONE:
                    if dis_map[u][v] > MAX_LINE_DISTANCE_CROSS:
                        break
                if sign_map[u][v] == LOOP_ZONE:
                    if dis_map[u][v] > MAX_LOOP_DISTANCE_CROSS:
                        break
            if is_path:
                # 提取链接的路径部分
                cross_path = cross_path & (~cross)
                for it in range(1,3):
                    # 有序的膨胀，测试链接点宽度是否过小
                    new_cross_path = cv2.dilate(cross_path.astype(np.uint8), np.ones((3,3),np.uint8), iterations=it).astype(np.bool8)
                    # 达到链接处宽度
                    if np.any(new_cross_path & (~ori_segment)):
                        # 将thinned砍断
                        thinned[cross_path] = 0
                        # 将angle_map去除
                        new_angle_map[cross_path] = -1
                        break
    return new_angle_map, thinned


def remove_cross_zone(angle_map, debug=False):
    """若交叉点不与loop相连，则不视为交叉点
    """
    new_angle_map = angle_map.copy()
    cross = np.where(angle_map == CROSS_SIGN)
    label = np.zeros_like(angle_map,np.uint8)
    h,w = label.shape
    # label标记是否访问，255为不需要访问
    for i in range(cross[0].shape[0]):
        # graph为BFS_path搜索的图
        graph = angle_map.copy()
        x,y = cross[0][i], cross[1][i]
        if label[x][y] == 255:
            continue
        # label[x][y] = 0
        # 将该点周围的同一cross区域均置label为255、置graph为FIRST_CROSS_SIGN
        local_cross = BFS(graph==CROSS_SIGN, (x,y), (0,0,0,0,0), direction=True) >= 0
        graph[local_cross] = FIRST_CROSS_SIGN
        label[local_cross] = 255
        
        sign_map, from_map, dis_map = BFS_path(graph, (x,y))
        loop_zone = BFS((graph==FIRST_CROSS_SIGN)|(sign_map==LOOP_ZONE), (x,y), (0,0,0,0,0), direction=True) >= 0
        if debug:
            visual = visualize_sign_map(sign_map, graph)
            cv2.imwrite(f"remove{i}.png",visual)
        # cross区域不和loop区域链接，说明该cross点只是三岔点，并非血管交点
        if np.count_nonzero(loop_zone) == np.count_nonzero(local_cross):
            new_angle_map[local_cross] = 1
        
    return new_angle_map
    

def BFS_path(graph, s):
    #s是起始点
    ''' 
    第二次进入-10区域时，不再访问出去
    -1 unvisited
    0 visited
    1 loop-point
    10 first cross
    100 second cross
    '''
    dis_map = np.full_like(graph.astype(int), -1)
    sign_map = np.full_like(graph.astype(int), -1)
    dx = [0,1,0,-1,1,-1,1,-1]
    dy = [1,0,-1,0,1,1,-1,-1]
    queue=[]
    queue.append(s)
    dis_map[s[0]][s[1]] = 0
    step = 0
    sign_map[s[0]][s[1]] = LINE_ZONE
    h,w = graph.shape
    from_map = [ [None for i in range(w)] for j in range(h)]
    cnt_map = np.zeros_like(graph.astype(int))
    while(len(queue)>0):
        vertex=queue.pop(0)
        step = dis_map[vertex[0]][vertex[1]] + 1
        cnt = cnt_map[vertex[0]][vertex[1]]
        for i in range(len(dx)):
            x = vertex[0] + dx[i]
            y = vertex[1] + dy[i]
            if x >= h or y >= w or x < 0 or y < 0:
                continue
            # 若不是mask范围内
            if graph[x][y] == -1:
                continue
            if cnt == 1 and graph[x][y] >= 0:
                continue
            # 已经访问过
            if sign_map[x][y] != -1:
                # 防止三角形的小闭环
                if from_map[x][y] is not None:
                    if abs(from_map[x][y][0] - from_map[vertex[0]][vertex[1]][0]) > 1 or abs(from_map[x][y][1] - from_map[vertex[0]][vertex[1]][1]) > 1:
                        sign_map[x][y] = LOOP_ZONE
                        u,v = vertex
                        while from_map[u][v] is not None:
                            if sign_map[u][v] != CROSS_ZONE:
                                sign_map[u][v] = LOOP_ZONE
                            u,v = from_map[u][v]
                        u,v = x,y
                        while from_map[u][v] is not None:
                            if sign_map[u][v] != CROSS_ZONE:
                                sign_map[u][v] = LOOP_ZONE
                            u,v = from_map[u][v]
                continue
            # 再次进入cross
            cnt_map[x][y] = cnt
            if graph[x][y] == CROSS_SIGN and graph[vertex[0]][vertex[1]] >= 0:
                cnt_map[x][y] = cnt + 1
            sign_map[x][y] = LINE_ZONE
            if graph[x][y] == CROSS_SIGN:
                sign_map[x][y] = CROSS_ZONE
            from_map[x][y] = vertex
            dis_map[x][y] = step
            queue.append((x,y))
    return sign_map, from_map, dis_map


def lookup_angle(point, angle_field):
    a = 3
    h = point[0]
    w = point[1]
    sample = angle_field[h-a:h+a,w-a:w+a]
    if np.any(sample > 0):
        return np.mean(sample[sample > 0])
    if np.any(sample == CROSS_SIGN):
        return CROSS_SIGN
    return -1

# ADD: 在分叉处选择切向尽量连续的分叉
def smart_BFS(graph, s, line, angle_field):  
    """对mask进行bfs，标注从起点开始的dis_map,且在分割直线的右侧

    Args:
        graph (np.array(bool)): m*n 
        s (tuple): 起点坐标
        line=(a,b,c,up,down): 分界线 ax+by < c down<y<up
        angle_field (np.array(int)): 方向场
    Returns:
        dis_map: 每个点到起点的距离
    """
    # 角度衰减率
    DECAY = 0.2
    # 最大角度偏移
    ANGLE_RANGE = 30
    #s是起始点
    a,b,c,up,down = line
    dis_map = np.full_like(graph.astype(int), -1)
    # 路线的累积平均角度
    mean_angle_map = np.full_like(graph.astype(int), -1)
    dx = [0,1,0,-1]#,1,-1,1,-1]
    dy = [1,0,-1,0]#,1,1,-1,-1]
    queue=[]
    queue.append(s)
    dis_map[s[0]][s[1]] = 0
    mean_angle_map[s[0]][s[1]] = lookup_angle(s, angle_field)
    step = 0
    h,w = graph.shape
    while(len(queue)>0):
        vertex=queue.pop(0)
        step = dis_map[vertex[0]][vertex[1]] + 1
        path_angle = mean_angle_map[vertex[0]][vertex[1]]
        last_angle = lookup_angle(vertex, angle_field)
        for i in range(len(dx)):
            x = vertex[0] + dx[i]
            y = vertex[1] + dy[i]
            current_angle = lookup_angle((x,y),angle_field)
            if x >= h - BFS_PADDING or y >= w - BFS_PADDING or x < BFS_PADDING or y < BFS_PADDING:
                continue
            # 若不是mask范围内
            if graph[x][y] == False:
                continue
            # 已经访问过
            if dis_map[x][y] != -1:
                continue
            if a*x + b*y > c and y > down and y < up:
                continue
            # 若来自交叉点附近
            if last_angle == CROSS_SIGN and current_angle > 0 and abs(current_angle - path_angle) > ANGLE_RANGE:
                continue
            dis_map[x][y] = step
            if current_angle == CROSS_SIGN:
                mean_angle_map[x][y] = path_angle
            else:
                mean_angle_map[x][y] = path_angle * DECAY + current_angle * (1 - DECAY)
            queue.append((x,y))
    return dis_map


def BFS(graph, s, line, direction=False):  
    """对mask进行bfs，标注从起点开始的dis_map,且在分割直线的右侧

    Args:
        graph (np.array(bool)): m*n 
        s (tuple): 起点坐标
        line=(a,b,c,up,down): 分界线 ax+by < c down<y<up
    Returns:
        dis_map: 每个点到起点的距离
    """
    #s是起始点
    a,b,c,up,down = line
    dis_map = np.full_like(graph.astype(int), -1)
    dx = [0,1,0,-1]#,1,-1,1,-1]
    dy = [1,0,-1,0]#,1,1,-1,-1]
    if direction:
        dx = [0,1,0,-1,1,-1,1,-1]
        dy = [1,0,-1,0,1,1,-1,-1]
    queue=[]
    queue.append(s)
    dis_map[s[0]][s[1]] = 0
    step = 0
    h,w = graph.shape
    while(len(queue)>0):
        vertex=queue.pop(0)
        step = dis_map[vertex[0]][vertex[1]] + 1
        for i in range(len(dx)):
            x = vertex[0] + dx[i]
            y = vertex[1] + dy[i]
            if x >= h or y >= w or x < 0 or y < 0:
                continue
            # 若不是mask范围内
            if graph[x][y] == False:
                continue
            # 已经访问过
            if dis_map[x][y] != -1:
                continue
            if a*x + b*y > c and y > down and y < up:
                continue
            dis_map[x][y] = step
            queue.append((x,y))
    return dis_map


def get_dis_map(x, y, binarysegment):
    # 计算dis_map
    a = x[1] - y[1]
    b = y[0] - x[0]
    c = y[0] * x[1] - y[1] * x[0]
    up = max(x[1],y[1]) + 10
    down = min(x[1],y[1]) - 10
    # 血管的分割
    graph = binarysegment.transpose()
    # 计算直线在分割上的中点
    length = 20
    start = None
    end = None
    for l in range(0,length):
        test = [ (l * x[0]+ (length - l) * y[0])/length , (l * x[1]+ (length - l) * y[1])/length ]
        if start is None and binarysegment[int(test[1])][int(test[0])]:
            start = test.copy()
        if start is not None and end is None and not binarysegment[int(test[1])][int(test[0])]:
            end  = test.copy()
    # 起点为顶端上下关键点中点
    s = [ int((start[0]+end[0])/2), int((start[1]+end[1])/2) ]
    sr = [s[0]+1, s[1]]
    if a > 0:
        right = BFS(graph, sr, (a,b,c,up,down))
        left = BFS(graph, s, (-a,-b,-c,up,down))
    else:
        left = BFS(graph, s, (a,b,c,up,down))
        right = BFS(graph, s, (-a,-b,-c,up,down))
    # 左边到顶端的距离为负，右边到顶端距离为正
    dis_map = np.full_like(graph.astype(int),-1)
    
    right[(right > left) & (left != -1)] = -1
    left[(left > right) & (right != -1)] = -1
    base = np.max(left)
    dis_map[left >= 0] = base - left[left >= 0]
    dis_map[right >= 0] = base + right[right >= 0]
    # 其他未标注距离但在mask中的位置为0
    # dis_map[dis_map==-1 & graph] = 0
    # 不在mask中的距离无穷大
    # dis_map[~graph] = 1e10
    # dis_map[dis_map==-1] = 0
    return dis_map.transpose()



def smart_get_dis_map(x, y, binarysegment, angle_field, visualize=None, debug=False):
    # 计算dis_map
    a = x[1] - y[1]
    b = y[0] - x[0]
    c = y[0] * x[1] - y[1] * x[0]
    up = max(x[1],y[1]) + 10
    down = min(x[1],y[1]) - 10
    # 血管的分割
    graph = binarysegment.transpose()
    angle_field = angle_field.transpose()
    # 计算直线在分割上的中点
    length = 20
    start = None
    end = None
    for l in range(0,length):
        test = [ (l * x[0]+ (length - l) * y[0])/length , (l * x[1]+ (length - l) * y[1])/length ]
        if start is None and binarysegment[int(test[1])][int(test[0])]:
            start = test.copy()
        if start is not None and end is None and not binarysegment[int(test[1])][int(test[0])]:
            end  = test.copy()
    # 起点为顶端上下关键点中点
    if start is None or end is None:
        raise Exception("NO SEG")
    s = [ int((start[0]+end[0])/2), int((start[1]+end[1])/2) ]
    sr = [s[0]+1, s[1]]
    if a > 0:
        right = smart_BFS(graph, sr, (a,b,c,up,down), angle_field)
        left = smart_BFS(graph, s, (-a,-b,-c,up,down), angle_field)
    else:
        left = smart_BFS(graph, sr, (a,b,c,up,down), angle_field)
        right = smart_BFS(graph, s, (-a,-b,-c,up,down), angle_field)
    # 左边到顶端的距离为负，右边到顶端距离为正
    dis_map = np.full_like(graph.astype(int),-1)
    if debug:
        right_visual = visualize(right.transpose(), binarysegment)
        left_visual = visualize(left.transpose(), binarysegment)
        cv2.imwrite("right.png", right_visual)
        cv2.imwrite("left.png", left_visual)
    # 将cross区域right枝置为-1
    cross = (angle_field==CROSS_SIGN)
    cross = cv2.dilate(cross.astype(np.uint8), np.ones((3,3),np.uint8), iterations=3).astype(np.bool8)
    right[cross] = -1
    # 
    right[(right > left) & (left != -1)] = -1
    left[(left > right) & (right != -1)] = -1
    base = np.max(left)
    dis_map[left >= 0] = base - left[left >= 0]
    dis_map[right >= 0] = base + right[right >= 0]
    # 其他未标注距离但在mask中的位置为0
    # dis_map[dis_map==-1 & graph] = 0
    # 不在mask中的距离无穷大
    # dis_map[~graph] = 1e10
    # dis_map[dis_map==-1] = 0
    return dis_map.transpose()