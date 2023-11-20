# 对识别点进行流速检测
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
import numpy as np

SPEED_VARIANCE = 40
FLOAT_RANGE = 5

def pyplot2numpy(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    data = np.asarray(buf)
    data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    return data


class WhiteCell:
    def __init__(self, frame_, id_, distance_=0):
        self.frame = frame_
        self.id = id_
        self.distance = distance_
        
    def toTuple(self):
        return (self.frame,self.id)
        

class WhiteCellFlow:
    """
    一条白细胞流
    """
    def __init__(self, start:WhiteCell):
        self.cells = [start]
        self.frames = [start.frame]
        pass
    
    def length(self):
        return len(self.cells)
    
    def top(self):
        return self.cells[len(self.cells)-1].toTuple()
    
    def append(self, next: WhiteCell):
        self.cells.append(next)
        self.frames.append(next.frame)
        
    def __add__(self, other):
        new = WhiteCellFlow(WhiteCell(0,0,0))
        new.cells = self.cells + other.cells
        new.frames = self.frames + other.frames
        return new

    def __eq__(self, other: WhiteCell) -> bool:
        if other.frame == self.frame and other.id == self.id:
            return True
        else:
            return False
        pass
    
    def get_variance(self):
        distance = [self.cells[i].distance-self.cells[i-1].distance for i in range(1,self.length())]
        distance = np.array(distance)
        return distance.max() - distance.min()
    
    def get_mean(self):
        distance = [ (self.cells[i].distance-self.cells[i-1].distance)/(self.frames[i] - self.frames[i-1]) for i in range(1,self.length())]
        distance = np.array(distance)
        distance.sort()
        if distance.shape[0] >= 5:
            return distance[2:-2].mean()
        elif distance.shape[0] >= 3:
            return distance[2]
        else:
            return distance[1]

    def get_var(self):
        distance = [ (self.cells[i].distance-self.cells[i-1].distance) for i in range(1,self.length())]
        distance = np.array(distance)
        return distance.var()
        
    def get_frame(self, start_frame, end_frame, all_distances, kp_length):
        start = None
        end = None
        for i,f in enumerate(self.frames):
            if start is None and f > start_frame:
                start = i
            if start is not None and f > end_frame:
                end = i
                break
        distance = [all_distances[kp_length[cell.frame]+cell.id] for cell in self.cells[start:end]]
        return self.frames[start:end], distance
    

class NailfoldVideo:
    def __init__(self, kps, dis_map, cell_range, distance_threshold=100, white_threshold=15):
        self.kps = kps
        self.dis_map = dis_map
        self.cell_range = cell_range
        self.distance_threshold = distance_threshold
        self.white_threshold = white_threshold
        kp_size = max([len(k) for k in self.kps])
        # 每个cell的上一帧对应cell
        self.all_frame_from_cell = [[None for _ in range(kp_size)] for _ in range(len(self.kps))]
        # 每个cell的下一帧对应cell
        self.all_frame_to_cell = [[None for _ in range(kp_size)] for _ in range(len(self.kps))]
        # 每个cell所在流的平均流速
        self.all_frame_speed = [[None for _ in range(kp_size)] for _ in range(len(self.kps))]
        
    def get_kp_vein_position(self, x):
        """获取识别点的“血管”位置

        Args:
            x (tuple): 识别点
            contours (list(list(p))): 图像中所有血管的骨架
        Returns:
            distance
        """
        h = int(x[1])
        w = int(x[0])
        # 防止取的包围盒中心点不在dis_map内，进行一个范围内的mean pooling
        a = 10
        while 1:
            core = self.dis_map[h-a:h+a,w-a:w+a]
            if np.any(core >= 0):
                break
            a += 10
        mean = np.mean(core[core >= 0])
        return mean
    
    def get_flow_length(self, cell: WhiteCell):
        start = cell
        start_len = 0
        end = cell
        end_len = 0
        while self.all_frame_from_cell[start.frame][start.id] is not None:
            start = self.all_frame_from_cell[start.frame][start.id]
            start_len += 1
        while self.all_frame_from_cell[end.frame][end.id] is not None:
            end = self.all_frame_from_cell[end.frame][end.id]
            end_len += 1
        return start_len + end_len + 1
    
    def get_frames_flow_direction(self):
        from sklearn.decomposition import PCA
        all_frame_direction = np.array([0 for _ in range(len(self.kps))])
        model = PCA()
        # TODO:
        return all_frame_direction

    def connect_cells(self, last, now, path):
        self.all_frame_to_cell[last.frame][last.id] = now
        self.all_frame_from_cell[now.frame][now.id] = last
        self.all_frame_speed[now.frame][now.id] = path.get_mean()
        self.all_frame_speed[last.frame][last.id] = path.get_mean()
    
    def compare_trustable_flow(self, old_path, path):
        # 指数越小越好
        length_rate = abs(old_path.length() / path.length())
        speed_rate = abs(path.get_mean() / old_path.get_mean())
        smooth_rate = abs(path.get_var() / old_path.get_var())
        if length_rate * speed_rate < 1 and smooth_rate < 1.5:
            return True
        return False

    def unify_all_flow(self, all_path):
        """将所有白细胞的路径归一成一致的路线

        Args:
            all_path (_type_): _description_

        Returns:
            list( WhiteCellFlow ): all_flow
        """
        # 每帧的path ([(WhiteCell), ...])
        kp_size = max([len(k) for k in self.kps])
        # 每个cell的上一帧对应cell
        self.all_frame_from_cell = [[None for _ in range(kp_size)] for _ in range(len(self.kps))]
        # 每个cell的下一帧对应cell
        self.all_frame_to_cell = [[None for _ in range(kp_size)] for _ in range(len(self.kps))]
        # 每个cell所在流的平均流速
        self.all_frame_speed = [[None for _ in range(kp_size)] for _ in range(len(self.kps))]
        # 每帧的大致流速（方向）
        all_frame_direction = np.array([0 for _ in range(len(self.kps))])
        all_frame_best_flow = [None for i in range(len(self.kps))] # Flow
        # all_frame_count = np.array([0 for _ in range(len(self.kps))])
        # 每帧流向的判定是 流速最小 流的长度尽量长
        for path in all_path:
            # path (distance, frame, id) white_kps
            last = None
            for i, cell in enumerate(path.cells[1:]):
                distance = (cell.distance - path.cells[i].distance) / (cell.frame - path.cells[i].frame)
                if abs(distance) > FLOAT_RANGE:
                    old_path = all_frame_best_flow[cell.frame]
                    # 只保留来自一个cell的最佳的flow
                    if old_path is None or self.compare_trustable_flow(old_path, path):
                            all_frame_best_flow[cell.frame] = path
        for frame_id, flow in enumerate(all_frame_best_flow):
            if flow is not None:
                all_frame_direction[frame_id] = np.sign(flow.get_mean())
        # 纠正矛盾的流向：所选流途径的帧流向相反
        for i in range(all_frame_direction.shape[0]):
            if all_frame_best_flow[i] is None:
                continue
            start = all_frame_best_flow[i].frames[0]
            end = all_frame_best_flow[i].frames[-1]
            for j in range(start,end+1):
                if all_frame_best_flow[j] is None:
                    continue
                # 流向相反
                if all_frame_direction[i] * all_frame_direction[j] < 0:
                    if j < i:
                        all_frame_direction[i] = all_frame_direction[j]
                        break
                    # j帧的流方向，证据更强
                    if self.compare_trustable_flow(all_frame_best_flow[i], all_frame_best_flow[j]):
                        all_frame_direction[i] = all_frame_direction[j]
                        break

        # 去除孤立的流向
        

        it = tqdm(all_path)
        it.set_description("unify")
        for path in it:
            # path [WhiteCellFlow]
            last = None
            for cell in path.cells:
                if last is not None:
                    distance = (cell.distance - last.distance) / (cell.frame - last.frame)
                else:
                    distance = (path.cells[1].distance - cell.distance)  / (path.cells[1].frame - cell.frame)
                now = cell
                # 若该流和经过该点的其他流方向不同，跳过
                if all_frame_direction[now.frame] != 0 and np.sign(distance) != np.sign(all_frame_direction[now.frame]): 
                    last = now
                    continue
                # 若last is None则是起点，不操作
                # 否则检验是否连接
                if last is not None:
                    smooth = True
                    if self.all_frame_speed[last.frame][last.id] is not None:
                        if abs(distance - self.all_frame_speed[last.frame][last.id]) > SPEED_VARIANCE/2:
                            smooth = False
                    # 没来源或来源一致
                    if self.all_frame_from_cell[now.frame][now.id] is None:
                        # last没有目的或目的一致
                        if self.all_frame_to_cell[last.frame][last.id] is None:
                            if smooth:
                                self.connect_cells(last, now, path)
                        elif self.all_frame_to_cell[last.frame][last.id].toTuple() == now.toTuple():
                            pass
                        # last已经有另一目的
                        else:
                            old = self.all_frame_to_cell[last.frame][last.id]
                            origin_gap = old.frame - last.frame
                            new_gap = now.frame - last.frame
                            origin_error = abs(self.all_frame_speed[last.frame][last.id] - (old.distance - last.distance)/origin_gap)
                            new_error = abs(path.get_mean() - distance)
                            origin_length = self.get_flow_length(old)
                            new_length = path.length()
                            # now更适合之前的路径
                            if smooth and new_error < origin_error + 10 and origin_length < new_length * 2:
                                
                                self.all_frame_from_cell[old.frame][old.id] = None
                                old_speed = self.all_frame_speed[last.frame][last.id]
                                self.connect_cells(last, now, path)
                                # self.all_frame_to_cell[last.frame][last.id] = now
                                # self.all_frame_from_cell[now.frame][now.id] = last
                                self.all_frame_speed[now.frame][now.id] = (old_speed + path.get_mean())/2
                            # 否则视作新一条流的起点
                    elif self.all_frame_from_cell[now.frame][now.id].toTuple() == last.toTuple():
                        pass
                    # 有另一个不同来源
                    else:
                        old = self.all_frame_from_cell[now.frame][now.id]
                        origin_gap = now.frame - old.frame
                        new_gap = now.frame - last.frame
                        origin_error = abs(self.all_frame_speed[old.frame][old.id] - (now.distance - old.distance)/origin_gap)
                        new_error = abs(path.get_mean() - distance)
                        origin_length = self.get_flow_length(old)
                        new_length = path.length()
                        # now更适合新的来源
                        if smooth and new_error < origin_error + 10 and origin_length < new_length * 2:
                            self.all_frame_to_cell[old.frame][old.id] = None
                            self.connect_cells(last, now, path)
                        # 否则让path在此截断
                        # else:
                        #     break                        
                last = now
        
        all_flow = []
        for frame,frame_path in enumerate(self.all_frame_from_cell):
            for id, last in enumerate(frame_path):
                if last is None:
                    if self.all_frame_to_cell[frame][id] is not None:
                        distance = np.array(self.cell_range[frame][id]).mean()
                        all_flow.append(WhiteCellFlow(WhiteCell(frame,id,distance)))
                    continue
                is_in = False
                for flow in all_flow:
                    if flow.top() == (last.frame,last.id):
                        distance = np.array(self.cell_range[frame][id]).mean()
                        flow.append(WhiteCell(frame,id,distance))
                        is_in = True
                if not is_in:
                    distance = np.array(self.cell_range[frame][id]).mean()
                    all_flow.append(WhiteCellFlow(WhiteCell(frame,id,distance)))
                
                        
        # 截止到某frame的所有flow       
        new_all_flow = []
        for flow in all_flow:
            if len(flow.frames) < 3:
                continue
            new_all_flow.append(flow)
        return new_all_flow
        
    def visualize(self, new_all_flow):
        LENGTH = 30
        graphs = []
        all_distances = []
        sequence = np.array([])
        it = tqdm(enumerate(self.kps))
        it.set_description("kps->graph")

        kps_length = [0]
        all_cell_ranges = np.array([])
        for frame_id,frame in it:
            distances = np.array([self.get_kp_vein_position(p) for p in frame])
            cell_ranges = np.array([self.cell_range[frame_id][id] for id in range(len(frame))])
            kps_length.append(kps_length[frame_id]+len(distances))
            sequence = np.concatenate([sequence, np.full_like(distances, frame_id)])
            all_distances = np.concatenate([all_distances, distances])
            if all_cell_ranges.shape[0] == 0:
                all_cell_ranges = cell_ranges
            elif cell_ranges.shape[0] != 0:
                all_cell_ranges = np.concatenate([all_cell_ranges, cell_ranges],axis=0)
            fig = plt.figure(figsize=(3,3))
            plt.scatter(sequence[kps_length[max(frame_id-LENGTH,0)]:].flatten(), all_distances[kps_length[max(frame_id-LENGTH,0)]:], c='blue')
            # 画出该白细胞的上下界
            # if len(all_cell_ranges) > 0:
            #     plt.scatter(sequence[kps_length[max(frame_id-LENGTH,0)]:].flatten(), all_cell_ranges[kps_length[max(frame_id-LENGTH,0)]:,0], c='purple', marker='+')
            #     plt.scatter(sequence[kps_length[max(frame_id-LENGTH,0)]:].flatten(), all_cell_ranges[kps_length[max(frame_id-LENGTH,0)]:,1], c='green', marker='*')

            color = ['red', 'yellow', 'green']
            for i,flow in enumerate(new_all_flow):
                frames, distance = flow.get_frame(max(frame_id-LENGTH,0), frame_id, all_distances, kps_length)
                plt.plot(np.array(frames), np.array(distance), color = color[i%len(color)], linewidth=2.0)
            # path_distance = [distances[id] for id,last in all_frame_path[frame_id]]
            # all_path_distances += path_distance
            # path_length.append(path_length[frame_id] + len(path_distance))
            # path_sequence += np.full_like(np.array(path_distance), frame_id).tolist()
            # plt.scatter(np.array(path_sequence[path_length[max(frame_id-LENGTH,0)]:]).flatten(), np.array(all_path_distances[path_length[max(frame_id-LENGTH,0)]:]), c='b')
            if frame_id < LENGTH:
                plt.axis([0, LENGTH, 0, 450])
            else:
                plt.axis([frame_id-LENGTH, frame_id, 0, 450])
            plt.title(f"white_threshold:{self.white_threshold}")
            current = pyplot2numpy(fig)
            graphs.append(cv2.cvtColor(current,cv2.COLOR_RGB2BGR))
            plt.close(fig)
        return graphs

    def dfs_find_kp_flow(self, frame, id, distance_range=None, depth=10):
        """找该识别点的所有接下来所有的可能匹配点

        Args:
            frame (int): 第frame帧
            id (int): 第i个白细胞
            depth (int, optional): 搜索深度. Defaults to 1.
            
        Returns
            [ WhiteCellFlow, ...], 匹配路径的列表
        """
        p = self.kps[frame][id]
        
        first = WhiteCell(frame, id, self.get_kp_vein_position(p))
        # 下一帧、下两帧
        flow = WhiteCellFlow(first)
        result = []
        for frame_distance,f in enumerate(self.kps[frame+1:frame+3]):
            start_len = len(result)
            for i,kp in enumerate(f):
                next = WhiteCell(frame+frame_distance+1, i, self.get_kp_vein_position(kp))
                distance = (next.distance - first.distance) / (frame_distance+1)
                min_range, max_range = distance, distance
                if distance==0:
                    continue
                # 若上一层指定方向，则
                if distance_range is not None:
                    # # 只看流动方向相同的点
                    # if np.sign(direction) != np.sign(distance):
                    #     continue 
                    # 只保留变化小于一定值的
                    if abs(distance_range[0] - distance) > SPEED_VARIANCE:
                        continue
                    # 只保留变化小于一定值的
                    if abs(distance_range[1] - distance) > SPEED_VARIANCE:
                        continue
                    # 只允许在0附近浮动+-10
                    if distance_range[1] > FLOAT_RANGE and distance <= 0:
                        continue
                    if distance_range[0] < -FLOAT_RANGE and distance >= 0:
                        continue
                    min_range, max_range = min(distance_range[0], distance), max(distance_range[1], distance)
                # 只查看血管距离小于阈值的点，看其是否可能是该识别点在下一帧的对应点
                if abs(distance) < self.distance_threshold:
                    # 下一帧的识别点是否有位移匹配点
                    if depth == 0:
                        result = [flow]
                        return result
                    next_result = self.dfs_find_kp_flow(frame+1+frame_distance, i, distance_range=[min_range, max_range], depth=depth-1)
                    result += [ flow + line for line in next_result]
            # 若相隔一帧的已经找到，则不再找隔两帧的
            if len(result)-start_len > 0:
                break
        if len(result) == 0:
            result = [flow]
        return result
    
    def find_best_flow(self, frame, id, depth):
        """检验第frame帧的id号白细胞是否可追踪
        两次同正负的血管距离

        Args:
            frame (_type_): 第frame帧
            id (_type_): id号白细胞
        """
        results = self.dfs_find_kp_flow(frame, id, depth=8)
        best_path = []
        min_dev = SPEED_VARIANCE
        maxlen = 0
        for path in results:
            if path.length() < depth:
                continue
            # dev = path.get_variance()
            # if dev > min_dev:
            #     continue
            best_path.append(path)
        return best_path
     
    def find_all_flow(self, depth=4):
        all_path = []
        # kps = self.kps
        it = tqdm(enumerate(self.kps))
        it.set_description("dfs")
        for f,frame in it:
            for id,p in enumerate(frame):
                path = self.find_best_flow(f, id, depth = depth)
                all_path += path
        return all_path
      