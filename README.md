# WRCAN-PyTorch

The official implementation of WRCAN (Wide Receptive Field and Channel Attention Network for Deblurring of JPEG compressed Image)[[pdf]](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Lee_Wide_Receptive_Field_and_Channel_Attention_Network_for_JPEG_Compressed_CVPRW_2021_paper.pdf), which is ranked in the **3rd**, in the NTIRE 2021 challenge on Image Deblurring Track 2 JPEG artifacts.

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
<!-- paper 정보 추가 (TBA) -->

```bib
WRCAN
@InProceedings{Lee_2021_CVPR,
    author    = {Lee, Donghyeon and Lee, Chulhee and Kim, Taesung},
    title     = {Wide Receptive Field and Channel Attention Network for JPEG Compressed Image Deblurring},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {304-313}
}

EDSR
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```

## Dependencies
This source code is base on the PyTorch implementation of EDSR. 
Therefore, WRCAN-PyTorch follows the dependencies of the implementation of EDSR. 
For more information, please refer to [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).

* Python 3.8
* PyTorch >= 1.7
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Testing REDS dataset for deblurring JPEG
Place unzipped REDS dataset test images in ``--dir_data`` folder (like ``<dir_data>/REDS/test/test_blur_jpeg/<sequence_name>/<image_name>``). 

For Deblur Track2 in NTIRE2021, you can download REDS dataset using this script [here](https://gist.github.com/SeungjunNah/b10d369b92840cb8dd2118dd4f41d643).
Pre-trained model, which produces the results submitted to NTIRE challenge, can be downloaded from [model_cvprw21](https://www.dropbox.com/s/kz7tppzspgkhzux/model_best_patch320_epoch374.pt?dl=0)
The inference result images on the validation set and the test set can be downloaded from [val_results](https://www.dropbox.com/s/vqgzqunxn13ulnv/submission-REDS_VAL_20210320_08-42.tar.gz?dl=0) and [test_results](https://www.dropbox.com/s/t4iwefp71bobqbv/submission-REDS_TEST_20210320_08-59.tar.gz?dl=0), respectively.

### Using Jupyter notebook
Run the ``main.ipynb`` in the ``src`` folder. 
Before running the notebook file, please uncomment the options below ```"Test options"``` in the second code block and set the ``args.data_test`` appropriately.
```python
args.n_GPUs = 4
args.model = 'DeblurUNet'
args.act = 'relu'
args.scale = '1'
args.rgb_range = 1
args.n_depth = 2
args.n_resgroups = 4
args.n_resblocks = 16
args.data_test = 'REDS_TEST' # 'REDS_TEST' is for the testset, 'REDS_VAL' is for the validation set.
args.save = 'deblur_jpeg/wrcan'
args.precision = 'amp'
args.chop = True

# Test options
args.reset = False
args.data_range = '1-3000'
args.test_only = True
args.pre_train = os.path.join('../experiment', args.save, 'model/model_best_patch320_epoch374.pt')
args.save_results = True
args.self_ensemble = True
```

### Using command line
Run the script in ``src`` folder. 
Before you run the deblur, please uncomment the appropriate line in ```deblur.sh``` that you want to execute.
```bash
cd src       # You are now in */WRCAN-PyTorch/src
sh deblur.sh
```
## Training REDS dataset for deblurring JPEG
We trained WRCAN using jupyter notebook. 
But, training the model using command line is also possible by setting appropriate arguments. 

### Generate JPEG-compressed HR images.
With ``src/data/reds_gen_jpeg.py``, JPEG-compressed HR images are generated.

### Using Jupter notebook
Run the ``main.ipynb`` in the ``src`` folder. 
Please comment test options in the second block and set appropriate options for training. An example is described below.
```python
args.n_GPUs = 4
args.model = 'DeblurUNet'
args.act = 'relu'
args.scale = '1'
args.rgb_range = 1
args.n_depth = 2
args.n_resgroups = 4
args.n_resblocks = 16
args.data_train = 'REDS'
args.data_test = 'REDS_VAL'
args.data_range = '1-24000/1-100'
args.epochs = 400
args.loss = '1*L1+0.1*AE'
args.patch_size = 320
args.save = 'deblur_jpeg/wrcan'
args.save_results = False
args.reset = False
args.ch_shuffle = True
args.lr_noise_sigma = 2
args.decay = '100-200-250-300-350'
args.precision = 'amp'
args.chop = True
args.weight_decay = 1e-8
```
