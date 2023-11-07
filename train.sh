# python main.py --base dental_autoencoder.yaml -t --gpus 0,1,2,3,


python main.py --base dental_autoencoder_stage2_no_stage1.yaml -t --gpus 0,1,2,3,



python main.py --base dental_autoencoder_stage1.yaml -t --gpus 0,1,2,3,

python main.py --base dental_autoencoder_stage2_v2.yaml -t --gpus 0,1,2,3,


python main.py --base mri_autoencoder_stage1_registration_2.yaml -t --gpus 0,1,
python main.py --base mri_autoencoder_stage2_registration_2.yaml -t --gpus 0,1,2,3,


python main.py --base mri_autoencoder_stage2_registration_2.yaml -t --gpus 0,1,2,3,


python main.py --base mri_autoencoder_stage2_registration_Landmark_Image.yaml -t --gpus 0,1,2,3,4,5,6,7,



python main.py --base stage1/stage1_64.yaml -t --gpus 0,1,2,3,



python main.py --resume logs/_stage2_pre_2 --base stage2/experiment2/stage2_pre_2.yaml -t --gpus 0,1,

python main.py --resume logs/_stage2_pre_movement_2 --base stage2/experiment2/stage2_pre_movement_2.yaml -t --gpus 2,3,


python main.py --resume logs/_stage2_pre_movement_landmark_2 --base stage2/experiment2/stage2_pre_movement_landmark_2.yaml -t --gpus 2,3,

python main.py --resume logs/_stage2_pre_movement_landmark_line_2 --base stage2/experiment2/stage2_pre_movement_landmark_line_2.yaml -t --gpus 0,1,



python main.py --resume logs/_stage2_pre_movement_3 --base stage2/experiment3/stage2_pre_movement_3.yaml -t --gpus 0,
python main.py --resume logs/_stage2_pre_movement_landmark_3 --base stage2/experiment3/stage2_pre_movement_landmark_3.yaml -t --gpus 1,
python main.py --resume logs/_stage2_pre_movement_line_3 --base stage2/experiment3/stage2_pre_movement_line_3.yaml -t --gpus 6,
python main.py --resume logs/_stage2_pre_movement_landmark_line_3 --base stage2/experiment3/stage2_pre_movement_landmark_line_3.yaml -t --gpus 7,

