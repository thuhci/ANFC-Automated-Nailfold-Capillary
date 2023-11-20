import numpy as np
import cv2
import tqdm

def trans_affine(source, trans):
    """将图片平移trans

    Args:
        source (_type_): _description_
        trans (np.array (2,1)): _description_

    Returns:
        _type_: 平移后图片
    """
    rows, cols, _ = source.shape
    M = np.eye(2)
    M = np.concatenate([M,trans],axis=-1)
    # Warp the source image
    if source is not None:
        warp = cv2.warpAffine(source.copy(), M, (cols, rows))
    return warp


def imgs2imgs_by_trans(imgs, trans_list, debug=False):
    """根据偏移量将一组照片重新精确对齐

    Args:
        imgs (_type_): _description_
        trans_list (_type_): _description_
        start (int, optional): _description_. Defaults to 0.
        end (int, optional): _description_. Defaults to -1.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    it = tqdm.tqdm(enumerate(imgs))
    it.set_description("Affine with trans")
    new_imgs = []
    for i,img in it:
        new_img = trans_affine(img, trans_list[i])
        new_imgs.append(new_img)
    return new_imgs    
   
