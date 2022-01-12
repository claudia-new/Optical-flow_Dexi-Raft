import flowiz as fz
import os
import glob
from PIL import Image
import numpy as np
files = glob.glob('kitti_submission/*.flo')
out_dir = 'visualize/kitti_test'
#files = glob.glob('sintel_submission/clean/market_1/*.flo')
#out_dir = 'visualize/optical flow'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
flow = sorted(files)
for flo in flow:
    frame_id = flo.split('/')[-1]
    img = fz.convert_from_file(flo)
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(out_dir, frame_id+'.png'))
