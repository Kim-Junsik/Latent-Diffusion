# python main.py --base dental_autoencoder.yaml -t --gpus 0,1,2,3,
cd ..


# # python test_for_evaluation.py --eta=1.0 --training_path='_pre'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --training_path='_stage2_pre_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --training_path='_stage2_pre_movement_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --training_path='_stage2_pre_movement_landmark_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --training_path='_stage2_pre_movement_landmark_line_2'


# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --external --eta=0.0 --training_path='_stage2_pre_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --external --eta=0.0 --training_path='_stage2_pre_movement_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --external --eta=0.0 --training_path='_stage2_pre_movement_landmark_2'

# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --eta=0.0 --training_path='_stage2_pre_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --eta=0.0 --training_path='_stage2_pre_movement_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --eta=0.0 --training_path='_stage2_pre_movement_landmark_2'

# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --external --eta=0.0 --training_path='_stage2_pre_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --external --eta=0.0 --training_path='_stage2_pre_movement_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --external --eta=0.0 --training_path='_stage2_pre_movement_landmark_2'


# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --training_path='_stage2_pre_movement_landmark_line_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --eta=0.0 --training_path='_stage2_pre_movement_landmark_line_2'


# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --external --eta=0.0 --training_path='_stage2_pre_movement_landmark_line_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --ddim_use_original_steps --external --eta=0.0 --training_path='_stage2_pre_movement_landmark_line_2'


CUDA_VISIBLE_DEVICES=0 python test_for_evaluation.py --eta=0.0 --ddim_step=200 --training_path='_stage2_pre_movement_3'
CUDA_VISIBLE_DEVICES=0 python test_for_evaluation.py --eta=0.0 --ddim_step=200 --training_path='_stage2_pre_movement_3' --external
CUDA_VISIBLE_DEVICES=0 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_3'
CUDA_VISIBLE_DEVICES=0 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_3' --external


# CUDA_VISIBLE_DEVICES=1 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_2'
# CUDA_VISIBLE_DEVICES=1 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --external --training_path='_stage2_pre_movement_2'


# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_landmark_2'
# CUDA_VISIBLE_DEVICES=2 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --external --training_path='_stage2_pre_movement_landmark_2'


# CUDA_VISIBLE_DEVICES=3 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --training_path='_stage2_pre_movement_landmark_line_2'
# CUDA_VISIBLE_DEVICES=3 python test_for_evaluation.py --eta=0.0 --ddim_step=500 --external --training_path='_stage2_pre_movement_landmark_line_2'

