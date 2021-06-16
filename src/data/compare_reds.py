import os
import os.path
import cv2
import util
import imageio
import shutil 
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from collections import defaultdict
from datetime import datetime

def calc_psnr_ssim(rst_path0, rst_path1, ref_path, mod, top=10):    
	# ref_path = os.path.expanduser(ref_path)
	names_hr = sorted(util._get_paths_from_images(ref_path))

	if mod > 1:
		names_hr_val = [fname for fname in names_hr if (int(fname.split('/')[-1].split('.')[0].split('_')[0])+1)%mod == 0]
	else:
		names_hr_val = names_hr
	names_sr_val0 = [fname.replace(ref_path, rst_path0).replace('.png', '_x1_SR.png') for fname in names_hr_val]
	names_sr_val1 = [fname.replace(ref_path, rst_path1).replace('.png', '_x1_SR.png') for fname in names_hr_val]

	psnrs, ssims = [], []
	for (hr_path, sr_path0, sr_path1) in tqdm(list(zip(names_hr_val, names_sr_val0, names_sr_val1)), ncols=80):
		# print(hr_path)
		hr_name = '/'.join(hr_path.split('/')[-2:])
		ref_img = imageio.imread(hr_path)
		res_img0, res_img1 = imageio.imread(sr_path0), imageio.imread(sr_path1)

		psnr0, psnr1 = peak_signal_noise_ratio(ref_img, res_img0), peak_signal_noise_ratio(ref_img, res_img1)
		# ssim0 = structural_similarity(ref_img, res_img0, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
		# ssim1 = structural_similarity(ref_img, res_img1, multichannel=True, gaussian_weights=True, use_sample_covariance=False)

		psnrs.append((hr_name, psnr0, psnr1, psnr1-psnr0))
		# ssims.append((hr_name, ssim0, ssim1, ssim1-ssim0))
	
	psnrs.sort(key=lambda x: x[-1], reverse=True)
	# ssims.sort(key=lambda x: x[-1], reverse=True)
	
	return psnrs[:top], ssims[:top]

def copy_rst_images(blur_path, rst_path0, rst_path1, ref_path, save_path, psnrs, ssims):
	# for psnrs
	cnt = 0
	for (name, _, _, _) in tqdm(psnrs, ncols=80):
		src_blur = os.path.join(blur_path, name.replace('.png', '.jpg'))
		src_hr = os.path.join(ref_path, name)
		src_rst0 = os.path.join(rst_path0, name.replace('.png', '_x1_SR.png'))
		src_rst1 = os.path.join(rst_path1, name.replace('.png', '_x1_SR.png'))

		dst_name = name.replace('/', '_')
		dst_blur = os.path.join(save_path, dst_name.replace('.png', '_blur_{}.jpg'.format(cnt)))
		dst_rst0 = os.path.join(save_path, dst_name.replace('.png', '_rst0_{}.png'.format(cnt)))
		dst_rst1 = os.path.join(save_path, dst_name.replace('.png', '_rst1_{}.png'.format(cnt)))
		dst_hr = os.path.join(save_path, dst_name.replace('.png', '_sharp_{}.png'.format(cnt)))

		shutil.copy(src_blur, dst_blur)
		shutil.copy(src_rst0, dst_rst0)
		shutil.copy(src_rst1, dst_rst1)
		shutil.copy(src_hr, dst_hr)
		cnt += 1

	return
				



def main(blur_path, rst_path0, rst_path1, ref_path, mod=10, top=20):
	"""
	"runtime", "other" should be described to make submission folder. (write_readme_file())
	"""
	# absolute path
	blur_path = os.path.expanduser(blur_path)
	rst_path0 = os.path.expanduser(rst_path0)
	rst_path1 = os.path.expanduser(rst_path1)
	ref_path = os.path.expanduser(ref_path)
	# Make output folder
	temp = rst_path1.split('/')
	rst_path_base = '/'.join(temp[:-1])
	rst_name_base = temp[-1].replace('results', 'compare')
	save_path = os.path.expanduser(os.path.join(rst_path_base, rst_name_base))
	current_time = datetime.today().strftime("%Y%m%d_%H-%M")  # YYYY/mm/dd HH:MM 형태의 시간 출력
	save_path += '_'+current_time
	print('Save Path: ', save_path)
	os.makedirs(save_path, exist_ok=True)

	# Calculate PSNR and SSIM
	psnr_result, ssim_result = calc_psnr_ssim(rst_path0, rst_path1, ref_path, mod, top)
	# Write readme file (Write some explanation about this submission to the argument 'other')
# 	print(psnr_result)
# 	print(ssim_result)

	# Copy result files to save file
	copy_rst_images(blur_path, rst_path0, rst_path1, ref_path, save_path, psnr_result, ssim_result)
# 	copy_rst_images(rst_path, save_path, mod)

if __name__ == '__main__':
	"""
	blur_path
	rst_path0: the path of deblurred result images.
	rst_path1:
	ref_path: the path of ground truth images
	"""
	main(blur_path = '~/Datasets/REDS/val/val_blur_jpeg',
			ref_path='~/Datasets/REDS/val/val_sharp', 
			rst_path0='~/EDSR-PyTorch/experiment/deblur_jpeg/deblurnet_wabbody_ca_d2g4r16f64_re_chshuffle_noise_patch256/results-REDS_VAL_selfensemble', 
			rst_path1='~/WRCAN-PyTorch/experiment/deblur_jpeg/deblurnet_wabbody_ca_d2g4r16f64_re_chshuffle_noise_patch256_0.1L2aeloss_wd1e-8/results-REDS_VAL_selfensemble',
			mod=0,
            top=100)