# Inference
python main.py --model DeblurUNet --n_GPUs 4 --scale 1 --n_depth 2 --n_resgroups 4 --n_resblocks 16 --rgb_range 1 --data_test REDS_VAL --data_range 1-3000 --save deblur_jpeg/wrcan --pre_train ../experiment/deblur_jpeg/wrcan/model/model_best_patch320_epoch374.pt --test_only --chop --self_ensemble --save_results;

