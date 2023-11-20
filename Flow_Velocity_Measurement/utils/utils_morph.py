import os
from math import floor

import cv2
import numpy as np


class ImageSegmentation():
    ''' methods about image segmentation
    '''
    def __init__(self, img_seg):
        if img_seg.max() <= 1:
            img_seg = img_seg*255
        self.img_seg = img_seg
        self.img_thinning = self.get_img_thinning(img_seg)
        self.blk_h = 3
        self.blk_w = 15


    #### skeletonization
    def get_img_thinning(self, img_seg):
        '''skeletonization 
        '''
        kernel = np.ones((3, 1), np.uint8)
        img_bin = cv2.dilate(img_seg, kernel, iterations=1)
        img_bin = cv2.erode(img_bin, kernel, iterations=1)
        # img_bin = cv2.dilate(img_bin, kernel, iterations=1)
        img_thinning = cv2.ximgproc.thinning(img_bin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        # delete points in the edge because they are not reliable
        img_thinning[0,:] = 0
        img_thinning[-1,:] = 0
        img_thinning[:,0] = 0
        img_thinning[:,-1] = 0
        return img_thinning

    #### 8-connected graph component
    def get_neighbors(self, img_thinning, node, prev):
        '''
        input np.array img_thinning, tuple node(absolte coordinate), tuple prev
        return
        neighbor_ls: 8-connected neighbor list, except the prev neighbor
        point_class: str, 'junction' or 'endpoint' or 'normal' or 'useless'
        '''
        neighbor = np.where(img_thinning[max(0,node[0]-1):min(node[0]+2,img_thinning.shape[0]),max(0,node[1]-1):min(node[1]+2,img_thinning.shape[1])]==255) 

        x = neighbor[0]
        y = neighbor[1]
        neighbor_ls = [(x[i]+max(0,node[0]-1),y[i]+max(0,node[1]-1)) for i in range(len(x))] # (1,-1) means left down

        neighbor_ls.remove(node)
        point_class = self._get_node_class(neighbor_ls, node)
        if prev is not None and prev in neighbor_ls:
            neighbor_ls.remove(prev)
        return neighbor_ls, point_class


    def _get_node_class(self, neighbor_ls, node):
        node_class = ['endpoint','junction','normal','useless']
        count = 0
        rank = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]

        if len(neighbor_ls) == 1:
            # endpoint
            return node_class[0]
        elif len(neighbor_ls) == 2:
            # normal or useless
            for i in range(len(rank)):
                if (rank[i][0]+node[0],rank[i][1]+node[1]) in neighbor_ls and (rank[(2+i)%len(rank)][0]+node[0],rank[(2+i)%len(rank)][1]+node[1]) in neighbor_ls:
                    # useless(c==2)
                    return node_class[3]
                if (rank[i][0]+node[0],rank[i][1]+node[1]) in neighbor_ls and (rank[(1+i)%len(rank)][0]+node[0],rank[(1+i)%len(rank)][1]+node[1]) not in neighbor_ls:
                    count += 1
            # c==1: useless, c==2: normal
            return node_class[(count == 1)+2]
        else:
            for i in range(len(rank)):
                if (rank[i][0]+node[0],rank[i][1]+node[1]) in neighbor_ls and (rank[(1+i)%len(rank)][0]+node[0],rank[(1+i)%len(rank)][1]+node[1]) not in neighbor_ls:
                    count += 1
            # neight>2 and count == 2: normal
            return node_class[(count < 3)+1]


    def get_conncomp(self, img_thinning, seed): #TBD
        '''
        get the 8-connected graph component from the seed in certain order
        input: 
        img_thinning: numpy array, biograph skeleton
        seed: tuple #TBD
        return:
        close_ls_new: list, points in the the 8-connected graph component in certain order
        endpoints: list
        junctions: list
        '''
        # DFS
        # from open pop one node
        # get all neighbors, add node to close
        # check in old, if not, add to open
        if seed is None:
            return [],[],[]
        seed = tuple(seed)
        if img_thinning[seed]:
            open_ls = [seed]
            close_ls = []
            useless_ls = []
            junctions = []
            endpoints = []

            # check
            prev = None
            while open_ls:
                next = open_ls.pop()
                close_ls.append(next)
                neighbors, point_class = self.get_neighbors(img_thinning, next, prev)
                if point_class == 'useless':
                    useless_ls.append(next)
                    close_ls.pop(-1)
                    img_thinning[next] = 0
                elif point_class == 'junction':
                    junctions.append(next)
                elif point_class == 'endpoint':
                    endpoints.append(next)
                prev = next
                
                for neighbor in neighbors:
                    if neighbor not in close_ls and neighbor not in useless_ls:
                        if neighbor in open_ls:
                            open_ls.remove(neighbor)
                        open_ls.append(neighbor)
            
            # merge
            idx1 = close_ls.index(endpoints[0]) # first endpoint
            close_ls_new = close_ls[idx1::-1]+close_ls[idx1+1:]
            
            # clean for 2 times
            # delete the points with dist > 2+j
            for j in range(2):
                dist = np.zeros(len(close_ls_new))
                for i in range(len(close_ls_new)-1):
                    dist[i] = np.abs(close_ls_new[i][0]-close_ls_new[i+1][0])+np.abs(close_ls_new[i][1]-close_ls_new[i+1][1])
                close_ls_clean = [close_ls_new[i+1] for i in range(len(close_ls_new)-1) if dist[i]<=2*j+2]
                # self.draw_maps(close_ls_clean,2+j)
                close_ls_new = close_ls_clean
            return close_ls_clean,endpoints,junctions
        else:
            return [],[],[]


    def draw_maps(self, close_ls, output_dir, name=1):
        
        # visualize the close_ls
        close_ls_x = [close_ls[i][0] for i in range(len(close_ls))]
        close_ls_y = [close_ls[i][1] for i in range(len(close_ls))]
        mtx = np.zeros([-min(close_ls_x)+max(close_ls_x)+1,-min(close_ls_y)+max(close_ls_y)+1])
        for i in range(len(close_ls)):
            mtx[close_ls[i][0]-min(close_ls_x),close_ls[i][1]-min(close_ls_y)] = i
        
        # use heatmap to visualize the close_ls
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.heatmap(mtx)

        # save
        cv2.imwrite(os.path.join(output_dir,f"mtx_{name}.png"), mtx)
        print(f"save mtx at {os.path.join(output_dir,f'mtx_{name}.png')}")
        plt.close()
        return mtx


    def get_one_skeleton(self, img_thinning, x_range = [103,150], y_range = [500, 550]):
        '''
        get the single capillary in certain areas
        input:
        img_thinning: numpy array, biograph skeleton
        range: list, #TBD
        return:
        close_ls_new: list, points in the the 8-connected graph component in certain order
        endpoints: list
        junctions: list    
        '''
        # 假设只要一个连通域 #TBD
        seeds = np.where(img_thinning[x_range[0]:x_range[1],y_range[0]:y_range[1]]==255)
        x = seeds[0]
        y = seeds[1]
        seed = (x[0],y[0])
        close_ls_new,endpoints,junctions = self.get_conncomp(img_thinning,seed)
        return close_ls_new,endpoints,junctions


    #### auto-function
    def get_nearest_seed(self, img_thinning, pos):
        ''' find the nearest seed from given position
        '''
        a = 50 #TBD
        seeds = np.where(img_thinning[max(0,pos[0]-a):min(img_thinning.shape[0],pos[0]+a),max(0,pos[1]-a):min(img_thinning.shape[1],pos[1]+a)]==np.max(np.max(img_thinning))) #TBD 255 or 1
        x = seeds[0]
        y = seeds[1]
        if len(x) == 0:
            return None
        seeds_ls = [(x[i]+max(0,pos[0]-a),y[i]+max(0,pos[1]-a)) for i in range(len(x))]

        def dist(pos1,pos2):
            return (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2

        seeds_dist = []
        for seed in seeds_ls:
            seeds_dist.append(dist(seed, pos))
        nearest_seed = seeds_ls[seeds_dist.index(min(seeds_dist))]
        # print("min dist seed",min(seeds_dist), (nearest_seed[0],nearest_seed[1]))
        return (nearest_seed[0],nearest_seed[1])


    def auto_select_blocks(self, img_thinning, pos, num = 3, best_dist = 25):
        '''auto-generate some blocks(equals to have sections) per certain distance
        input:
        img_thinning: numpy array, biograph skeleton

        '''
        seed = self.get_nearest_seed(img_thinning, pos)
        connected_order,_,_ = self.get_conncomp(img_thinning,seed)

        # best_dist = 25 # 200
        # if best_dist is give: caculate blks_num according to best_dist,else caculate best_dist according to blks_num
        if best_dist == 0:
            best_dist = floor(len(connected_order)/num)
        else:
            num = floor(len(connected_order)/best_dist)
        seed_idx = [round(len(connected_order)-best_dist*(n+0.5)) for n in range(num)]
        blocks_pos = [connected_order[seed] for seed in seed_idx]
        print(f"select block{len(connected_order)} {best_dist} {num}")
        return blocks_pos, seed_idx


    def get_tangent(self, seed, num = 10):
        # BFS
        seed = tuple(seed)
        if self.img_thinning[seed]:
            open_ls = [seed]
            close_ls = []

            prev = None
            while open_ls and len(close_ls) <= num:
                next = open_ls.pop(0)
                close_ls.append(next)
                neighbors, point_class = self.get_neighbors(self.img_thinning, next, prev)
                prev = next
                
                for neighbor in neighbors:
                    if neighbor not in close_ls and neighbor not in open_ls:
                        open_ls.append(neighbor)
            
            tangent_ori = (close_ls[-1][0]-close_ls[-2][0],close_ls[-1][1]-close_ls[-2][1])
            return tangent_ori
        else:
            return None


    def get_cross_section(self, seed):
        if self.img_seg[tuple(seed)]:
            tangent_ori = self.get_tangent(seed)
            print("tangent_ori:",tangent_ori)
            if tangent_ori is None:
                return None, []
            k1 = -tangent_ori[1]/np.sqrt(tangent_ori[0]**2+tangent_ori[1]**2)
            k2 = tangent_ori[0]/np.sqrt(tangent_ori[0]**2+tangent_ori[1]**2)
            line = []
            cnt = [1,1]
            for r in range(20):
                p1,p2 = (round(seed[0]+r*k1),round(seed[1]+r*k2)), (round(seed[0]-r*k1),round(seed[1]-r*k2))
                if cnt[0] and self.img_seg[p1]:
                    line += [p1]
                else:
                    cnt[0] = 0
                if cnt[1] and self.img_seg[p2]:
                    line += [p2]
                else:
                    cnt[1] = 0
            return (k1,k2), line
        else:
            return None, []


    #### save segmentation
    def save_seg(self, output_dir, idx):
        '''
        save the segmentation result
        input:
        mask: numpy array, biograph skeleton, 0 or 1
        '''
        mask = self.img_seg
        mask3d = np.zeros([mask.shape[0],mask.shape[1],3])
        mask3d[:,:,0] = mask
        mask3d[:,:,1] = mask
        mask3d[:,:,2] = mask
        cv2.imwrite(os.path.join(output_dir,f"seg_{idx}.png"), mask3d)
        print(f"save seg at {os.path.join(output_dir,f'seg_{idx}.png')}")

    def save_thinning(self, output_dir, idx):
        '''
        save the segmentation result
        input:
        mask: numpy array, biograph skeleton, 0 or 1
        '''
        img_thinning = self.img_thinning
        mask3d = np.zeros([img_thinning.shape[0],img_thinning.shape[1],3])
        mask3d[:,:,0] = img_thinning
        mask3d[:,:,1] = img_thinning
        mask3d[:,:,2] = img_thinning
        cv2.imwrite(os.path.join(output_dir,f"thinning_{idx}.png"), mask3d)
        print(f"save thinning at {os.path.join(output_dir,f'thinning_{idx}.png')}")


    def get_average(self, img_org, connected):
        avg = np.zeros([3])
        for cn in connected:
            avg += img_org[cn[0],cn[1],:]
        return avg/len(connected)


    def get_block_average(self, img_org, blk_seed):
        w2 = int(self.blk_w/2)
        h2 = int(self.blk_h/2)
        block = img_org[blk_seed[0]-h2:blk_seed[0]+h2,blk_seed[1]-w2:blk_seed[1]+w2,1]
        return np.mean(np.mean(block))


    def get_block(self, blk_seed):
        w2 = int(self.blk_w/2)
        h2 = int(self.blk_h/2)
        block = [(x,blk_seed[1]-w2) for x in range(blk_seed[0]-h2,blk_seed[0]+h2)]
        block.extend([(x,blk_seed[1]+w2) for x in range(blk_seed[0]-h2,blk_seed[0]+h2)])
        block.extend([(blk_seed[0]-h2,x) for x in range(blk_seed[1]-w2,blk_seed[1]+w2)])
        block.extend([(blk_seed[0]+h2,x) for x in range(blk_seed[1]-w2,blk_seed[1]+w2)])
        return block


    def get_profile(self, img_org, conncomp):
        profile = np.zeros(len(conncomp))
        a = 3
        for i in range(len(conncomp)):
            # only G channel -> G-R
            # TODO: only use the pixels of the img_seg
            valid_area = np.zeros_like(img_org[:,:,1])
            valid_area[conncomp[i][0]-a:conncomp[i][0]+a+1,conncomp[i][1]-a:conncomp[i][1]+a+1] = 1
            valid_area = self.img_seg/255*valid_area
            profile[i] = np.mean(np.mean(img_org[:,:,1]*valid_area))
        return profile


def get_block(blk_seed, blk_w = 15, blk_h = 10):
    w2 = int(blk_w/2)
    h2 = int(blk_h/2)
    block = [(x,blk_seed[1]-w2) for x in range(blk_seed[0]-h2,blk_seed[0]+h2)]
    block.extend([(x,blk_seed[1]+w2) for x in range(blk_seed[0]-h2,blk_seed[0]+h2)])
    block.extend([(blk_seed[0]-h2,x) for x in range(blk_seed[1]-w2,blk_seed[1]+w2)])
    block.extend([(blk_seed[0]+h2,x) for x in range(blk_seed[1]-w2,blk_seed[1]+w2)])
    return block


def draw_line(annotated_image, x_bar, y_bar, k, color = (255,0,0)):    
    k1,k2 = k
    r = 10/np.sqrt(1+(k1/k2)**2)
    line_loc = [(round(y_bar-r*k2),round(x_bar-r*k1)),(round(y_bar+r*k2),round(x_bar+r*k1))] 
    line_width = 1        
    cv2.line(annotated_image, line_loc[0], line_loc[1], color, line_width)


def draw_point(annotated_image, connected, color_idx = 0):
    # print("draw_point:",len(connected))
    colors_old = ['b','g','r', 'c', 'm', 'y', 'blanchedalmond', 'blueviolet', 'brown', 'coral']
    colors_rgb = [(255,182,193),(250,128,114),(255,127,80),(128,0,0),(173,216,230),(70,130,180),(72,61,139),(186,85,211),(25,25,112),(0,206,209)]
    colors_name = ["LightPink","Salmon","Coral","Maroon","LightBLue","SteelBlue","DarkSlateBlue","MediumOrchid","MidnightBlue","DarkTurquoise"]
    for cn in connected:
        pt = (cn[1],cn[0])
        cv2.line(annotated_image, pt, pt, colors_rgb[color_idx%len(colors_rgb)], 1) # x y tranverse





if __name__ == '__main__':
    img_seg = cv2.imread("./Flow_Velocity_Measurement/18_best_seg.png", cv2.IMREAD_GRAYSCALE)
    img_org = cv2.imread("./Flow_Velocity_Measurement/18_best_seg.png", cv2.IMREAD_GRAYSCALE)
    output_dir = "./Flow_Velocity_Measurement"
    myseg = ImageSegmentation(img_seg)
    seed = myseg.get_nearest_seed(myseg.img_thinning, (150,100))
    t = myseg.get_tangent(seed)
    print(seed,t)

    close,endpoints,junctions = myseg.get_conncomp(myseg.img_thinning,seed)
    annotated_image = cv2.cvtColor(myseg.img_seg,cv2.COLOR_GRAY2BGR)
    draw_point(annotated_image, close)

    (k1,k2), line = myseg.get_cross_section(seed)

    draw_line(annotated_image, seed[0], seed[1], t)
    draw_point(annotated_image, line)
    # draw_point(annotated_image, [seed], 'yellow')
    # draw_point(annotated_image, junctions, 'yellow')

    import time
    time_tuple = time.localtime(time.time())
    cv2.imwrite(os.path.join(output_dir,f"{time_tuple[2]}_{time_tuple[4]}t10.png"), annotated_image) #np.concatenate([images,mask],axis=0))

    profile = myseg.get_profile(img_org, seed)

