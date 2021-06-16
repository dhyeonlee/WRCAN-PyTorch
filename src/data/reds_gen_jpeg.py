import os
import os.path
import cv2
import util
from tqdm import tqdm

def main(in_path, Q=25):
    # scan image file path
    names_hr = sorted(util._get_paths_from_images(os.path.expanduser(in_path)))
    base_name = os.path.expanduser(in_path).split('/')[-1]
    base_name_q = base_name+'Q{}'.format(Q)
    out_path_base = os.path.expanduser(in_path).replace(base_name, base_name_q)
    os.makedirs(out_path_base, exist_ok=True)
    
    # make out folders for sequences.
    out_folder_names = set(['/'.join(_.split('/')[:-1]).replace(base_name, base_name_q) for _ in names_hr])
    
    for o_folder in out_folder_names:
        os.makedirs(o_folder, exist_ok=True)
    # read each image and store it to corresponding path
#     for hr_path in names_hr:

    for hr_path in tqdm(names_hr, ncols=80):
        img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        # store it with jpeg compression
        o_path = hr_path.replace(base_name, base_name_q)
        o_path = o_path.replace('.png', '.jpg')
        cv2.imwrite(o_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), Q])
        
if __name__ == '__main__':
    
    main('~/Datasets/REDS/train/train_sharp', 25)