import flowiz as fz
import os
import glob
from PIL import Image
import numpy as np
#files = glob.glob('kitti_submission_train/*.flo')
#out_dir = 'visualize/kitti_train'
for dtype in ['clean', 'final']:
    files = sorted(glob.glob(os.path.join('sintel_submission_market', dtype, 'market_2', '*.flo')))
    out_dir = os.path.join('visualize/sintel_market',dtype)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(files)):
        
        img = fz.convert_from_file(files[i])
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(out_dir, (str(i)+'.png')))
