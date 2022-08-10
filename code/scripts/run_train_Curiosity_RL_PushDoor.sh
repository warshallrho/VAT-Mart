xvfb-run -a python td3_train_push_pull_door_curiosityDriven_mp.py \
    --primact_type pushing  \
    --action_type PushDoor \
    --out_dir /home/username/VAT_Mart/VAT_Data/ \
    --trial_id xxx  \
    --random_seed 61  \
    --RL_ckpt_dir /home/username/VAT_Mart/VAT_Data/RL_PushDoor_trial_id \
    --RL_load_epoch 5000  \
    --critic_model_version  model_3d_task_critic \
    --critic_exp_suffix exp_suffix \
    --critic_load_epoch 50    \
    --critic_feat_dim 128 \
    --batch_size 512  \
    --replay_buffer_size 2048  \
    --pos_range 0.4  \
    --explore_noise_scale 0.03 \
    --eval_noise_scale 0.03  \
    --noise_decay 1 \
    --decay_interval 500  \
    --guidance_reward 300  \
    --success_reward 500   \
    --q_lr 1e-4 \
    --policy_lr 1e-4  \
    --threshold 0.3 \
    --task_lower 10   \
    --task_upper 70   \
    --state_initial_position  \
    --state_joint_origins \
    --state_initial_dir \
    --use_random_up \
    --up_norm_thresh -0.866 \
    --pred_world_xyz 0 \
    --pred_residual_world_xyz 1 \
    --pred_residual_root_qpos 0 \
    --early_stop    \
    --wp_rot        \
    --state_axes_all \
    --critic_update_frequently  \
    --use_HER       \
    --HER_only_success  \
    --HER_only_attach   \
    --critic_score_threshold 0.5  \
    --critic_update_itr 1 \
    --critic_batch_size 32 \
    --coordinate_system world \
    --lbd_critic_penalty 500 \
    --num_processes 12     \
    --no_gui





