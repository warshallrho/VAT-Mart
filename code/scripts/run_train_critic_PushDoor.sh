python train_3d_task_critic.py \
    --exp_suffix xxx \
    --model_version model_3d_task_critic \
    --primact_type pushing \
    --action_type PushDoor \
    --offline_data_dir /home/username/VAT_Mart/VAT_Data/RL_PushDoor_trial_id/TRAIN1   \
    --offline_data_dir2 /home/username/VAT_Mart/VAT_Data/RL_PushDoorCUR_trial_id/TRAIN_CUR_500   \
    --offline_data_dir3 /home/username/VAT_Mart/VAT_Data/RL_PushDoorCUR_trial_id/TRAIN_CUR_1000   \
    --offline_data_dir4 /home/username/VAT_Mart/VAT_Data/RL_PushDoorCUR_trial_id/TRAIN_CUR_1500   \
    --offline_data_dir5 /home/username/VAT_Mart/VAT_Data/RL_PushDoorCUR_trial_id/TRAIN_CUR_2000   \
    --offline_data_dir6 /home/username/VAT_Mart/VAT_Data/RL_PushDoorCUR_trial_id/TRAIN_CUR_2500   \
    --offline_data_dir7 /home/username/VAT_Mart/VAT_Data/RL_PushDoorCUR_trial_id/TRAIN_CUR_3000   \
    --val_data_dir /home/username/VAT_Mart/VAT_Data/RL_PushDoor_trial_id/EVAL1 \
    --buffer_max_num 512  \
    --feat_dim 128   \
    --num_steps 5    \
    --batch_size 32  \
    --angle_system 1 \
    --num_train 30000 \
    --num_eval  30000  \
    --degree_lower 10  \
    --train_num_data_uplimit 720  \
    --val_num_data_uplimit 350  \
    --coordinate_system world \
    --sample_type fps  \


