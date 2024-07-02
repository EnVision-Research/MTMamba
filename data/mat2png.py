import os, glob, cv2, imageio
import scipy.io as sio
from skimage.morphology import thin
import numpy as np
from PIL import Image
import tqdm

gt_dir = '/dataset/baijionglin/dataset/PASCALContext/pascal-context/trainval'
img_dir = '/dataset/baijionglin/dataset/PASCALContext/JPEGImages'
save_dir = '/dataset/baijionglin/dataset/PASCALContext/edge'
ids = sorted([os.path.split(file)[-1] for file in glob.glob(os.path.join(gt_dir, "*.mat"))])

for i in tqdm.tqdm(ids):
    i = os.path.splitext(i)[0]
    # print(i)
    gt = os.path.join(gt_dir, "{}.mat".format(i))
    _tmp = sio.loadmat(gt)
    _edge = cv2.Laplacian(_tmp['LabelMap'], cv2.CV_64F)
    _edge = thin(np.abs(_edge) > 0).astype(np.float32)

    img_path = os.path.join(img_dir, "{}.jpg".format(i))
    _img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
    if _edge.shape != _img.shape[:2]:
        _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    _edge = np.array(_edge) * 255
    imageio.imwrite(os.path.join(save_dir, "{}.png".format(i)), _edge.astype(np.uint8))

    # break