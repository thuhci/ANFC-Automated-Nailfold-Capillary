import numpy as np
import cv2
class TangshanKp2Zhuhai():
    """
    将一个血管顶端的图片放置在一个t_w*t_h大小的白色画布的右上角
    适配珠海数据集的血管尺寸比例
    """
    def __init__(self, t_w=400, t_h=400, color_goal=110):
        # 目标缩放的长宽
        self.t_w = t_w
        self.t_h = t_h
        self.h = 0
        self.w = 0
        self.color_goal = color_goal
        pass
    
    def transform(self, image):
        hist = cv2.calcHist([image], [1], None, [256], [0, 256])
        main = np.where(hist > np.mean(hist))
        center_color = int(main[0].mean())
        sample = image.copy().astype(np.int16)
        sample += self.color_goal - center_color
        sample[sample < 0] = 0
        sample[sample > 255] = 255
        sample = sample.astype(np.uint8)
        self.h,self.w,_ = sample.shape
        blank = np.zeros((self.t_h,self.t_w,3), np.uint8)
        blank[:self.h,:self.w] = sample
        return blank
    
    def rev_transform(self, image):
        return image[:self.h,:self.w]
    
    
class Tangshan2Zhuhai():
    def __init__(self, scale=2, color_goal=110):
        # 目标缩放的长宽
        self.scale = scale
        self.h = None
        self.w = None
        self.GAP = 10
        self.color_goal = color_goal
        pass

    def get_size(self):
        return (self.scale+1)*(self.scale+1)
    
    def transform(self, image):
        """
        将一个大照片拆成若干小照片组成一个batch
        """
        sample = image.copy()

        self.h,self.w,_ = sample.shape
        samples = []
        self.start_p = []
        goal_h = int(self.h / self.scale)
        goal_w = int(self.w / self.scale)
        section_h = int((self.h * (self.scale-1) / self.scale) / (self.scale))
        section_w = int((self.w * (self.scale-1) / self.scale) / (self.scale))
        for i in range(self.scale+1):
            for j in range(self.scale+1):
                h,w = (i*section_h, j*section_w)
                self.start_p.append([w,h])
                samples.append(sample[h:h+goal_h, w:w+goal_w])

        new_samples = []
        for sample in samples:
            hist = cv2.calcHist([sample], [1], None, [256], [0, 256])
            main = np.where(hist > np.mean(hist))
            center_color = int(main[0].mean())
            # 更改色差
            sample = sample.astype(np.int16)
            sample += self.color_goal - center_color
            sample[sample < 0] = 0
            sample = sample.astype(np.uint8)
            new_samples.append(sample)
        return new_samples
    
    def mask_rev_transform(self, masks):
        """
        masks: np.array(np.bool8)
        """
        origin = np.zeros((self.h,self.w),np.bool8)
        goal_h = int(self.h / self.scale)
        goal_w = int(self.w / self.scale)
        section_h = int((self.h * (self.scale-1) / self.scale) / (self.scale))
        section_w = int((self.w * (self.scale-1) / self.scale) / (self.scale))
        cnt = 0
        for i in range(self.scale+1):
            for j in range(self.scale+1):
                h,w = (i*section_h, j*section_w)
                origin[h:h+goal_h, w:w+goal_w] = origin[h:h+goal_h, w:w+goal_w] | masks[cnt]
                cnt += 1
        return origin
    
    def kp_rev_transform(self, keypoints):
        new = []
        for i,kp in enumerate(keypoints):
            if kp.shape[0] == 0:
                continue
            new.append(kp + np.array(self.start_p[i]))
        if len(new) == 0:
            kps = []
        else:
            kps = np.concatenate(new)
        refine_kps = []
        for kp in kps:
            is_in = False
            for ref in refine_kps:
                if np.sum(np.abs(kp[0]-ref[0]))<self.GAP:
                    is_in = True
                    break
            if not is_in:
                refine_kps.append(kp)
        return np.array(refine_kps)
    
    def bbox_rev_transform(self, bboxs):
        new = []
        for i,bbox in enumerate(bboxs):
            if bbox.shape[0] == 0:
                continue
            new.append(bbox + np.array(self.start_p[i]+self.start_p[i]))
        if len(new) == 0:
            kps = []
        else:
            kps = np.concatenate(new)
        refine_kps = []
        for kp in kps:
            is_in = False
            for ref in refine_kps:
                if np.sum(np.abs(kp-ref)) < 4*self.GAP:
                    is_in = True
                    break
            if not is_in:
                refine_kps.append(kp)
        return refine_kps


class TangshanCrop2Zhuhai():
    """
    唐山数据截取200，200开始的中央区域
    """
    def __init__(self, t_w=600, t_h=400, padding=200, color_goal=110):
        # 目标缩放的长宽
        self.t_w = t_w
        self.t_h = t_h
        self.h = 0
        self.w = 0
        self.color_goal = color_goal
        self.padding = padding
        pass
    
    def transform(self, image):
        sample = image.copy().astype(np.int16)
        hist = cv2.calcHist([image], [1], None, [256], [0, 256])
        main = np.where(hist > np.mean(hist))
        center_color = int(main[0].mean())
        sample = image.copy().astype(np.int16)
        sample += self.color_goal - center_color
        sample[sample < 0] = 0
        sample[sample > 255] = 255
        sample = sample.astype(np.uint8)
        self.h,self.w,_ = sample.shape
        new = sample[self.padding:self.padding+self.t_h,self.padding:self.padding+self.t_w]
        return new
    
    def rev_transform(self, image):
        origin = np.zeros((self.h,self.w,3),np.uint8)
        origin[self.padding:self.padding+self.t_h,self.padding:self.padding+self.t_w] = image
        return origin
    
    def kp_rev_transform(self, keypoints):
        keypoints += self.padding
        return keypoints