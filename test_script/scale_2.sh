# CUDA_VISIBLE_DEVICES=3 python test_for_evaluation_scale_extrapolation.py --eta=0.0 --ddim_step=200 --training_path='_stage2_pre_movement_3'
CUDA_VISIBLE_DEVICES=3 python test_for_evaluation_scale_extrapolation.py --eta=0.0 --ddim_step=200 --training_path='_stage2_pre_movement_3' --external

CUDA_VISIBLE_DEVICES=3 python test_for_evaluation_scale_extrapolation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_3'
CUDA_VISIBLE_DEVICES=3 python test_for_evaluation_scale_extrapolation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_3' --external