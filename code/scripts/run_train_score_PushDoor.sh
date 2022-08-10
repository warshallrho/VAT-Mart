python train_3d_task_score.py \
    --exp_suffix xxx \
    --model_version model_3d_task_score \
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
    --actor_path /home/username/VAT_Mart/VAT_Data/exp-model_3d_task_actor-PushDoor-exp_suffix/ckpts/%s-network.pth \
    --critic_path /home/username/VAT_Mart/VAT_Data/exp-model_3d_task_critic-PushDoor-exp_suffix/ckpts/%s-network.pth \
    --buffer_max_num 512 \
    --feat_dim 128   \
    --num_steps 5    \
    --batch_size 32  \
    --angle_system 1 \
    --num_train 30000 \
    --num_eval  30000  \
    --degree_lower 10  \
    --actor_eval_epoch 110 \
    --critic_eval_epoch 50 \
    --train_num_data_uplimit 720  \
    --val_num_data_uplimit 350  \
    --topk 5





