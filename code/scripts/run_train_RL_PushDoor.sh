xvfb-run -a python RL_train_push_door_mp.py \
    --primact_type pushing  \
    --action_type PushDoor  \
    --out_dir /home/username/VAT_Mart/VAT_Data/ \
    --trial_id xxx \
    --batch_size 512  \
    --replay_buffer_size 2048  \
    --pos_range 0.4  \
    --explore_noise_scale 0.1 \
    --eval_noise_scale 0.1  \
    --noise_decay 0.5 \
    --decay_interval 500 \
    --guidance_reward 300.0 \
    --success_reward 500 \
    --q_lr 1e-4 \
    --policy_lr 1e-4  \
    --threshold 0.3 \
    --task_lower 10.0 \
    --task_upper 70.0 \
    --with_step 0 \
    --state_initial_position \
    --state_joint_origins  \
    --state_initial_dir \
    --use_random_up \
    --up_norm_thresh -0.866 \
    --pred_world_xyz 0 \
    --pred_residual_world_xyz 1 \
    --pred_residual_root_qpos 0 \
    --use_HER \
    --HER_only_success \
    --HER_only_attach \
    --HER_move_margin 0 \
    --early_stop \
    --state_axes_all \
    --wp_rot \
    --rot_degree 30.0 \
    --no_gui \
    --num_steps 4 \
    --task_succ_margin 0.10 \
    --all_shape \
    --num_processes 10
