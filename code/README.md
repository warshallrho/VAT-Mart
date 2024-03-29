# Experiments
This folder includes the codebase for VAT-Mart project.

## Before start
To train the models, please first go to the `../data` folder and download the pre-processed SAPIEN dataset for VAT-Mart. 

## Dependencies
This code has been tested on Ubuntu 18.04 with Cuda 10.1, Python 3.6, and PyTorch 1.7.0.

First, install SAPIEN following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl

For other Python versions, you can use one of the following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

Please do not use the default `pip install sapien` as SAPIEN is still being actively developed and updated.

Then, if you want to run the 3D experiment, this depends on PointNet++.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

Finally, run the following to install other packages.

    # make sure you are at the repository root directory
    pip install -r requirements.txt

to install the other dependencies.

For visualization, please install blender v2.79 and put the executable in your environment path.
Also, the prediction result can be visualized using MeshLab or the *RenderShape* tool in [Thea](https://github.com/sidch/thea).

If you want to run on a headless server, simple put `xvfb-run -a ` before any code command that runs the SAPIEN simulator.
Install the `xvfb` tool on your server if not installed.

## Training Pipeline for the VAT-Mart Framework 

We use the experiments for the task `push door` as an example, and the scripts for other tasks are similar.



### Train RL for Data Collection
To efficiently collect action trajectories for completing varying tasks across diverse shapes and contact points,
we train an RL policy using TD3.
In the RL training, 
since the RL policy is trained in the simulation for only collecting training data to supervise the perception networks,
we can have access to the ground-truth state information of the simulation environment, 
such as the part poses, joint axis, and gripper poses.
At the test time, 
we discard the RL network and only use the learned perception networks to predict the proposed actionable visual priors

    sh scripts/run_train_RL_PushDoor.sh


### Collect Data using Trained RL
We collect action trajectories using the trained RL policy.

    sh scripts/run_collect_PushDoor.sh

### Train Trajectory Scoring Module using Collected Data
We train the Trajectory Scoring Module using the previously collected data.
In the following curiosity-driven RL training process, 
this module will provide the curiosity-feedback for RL,
and will be reversely finetuned by diverse trajectories proposed by RL.

    sh scripts/run_train_critic_PushDoor_before.sh

### Train Curiosity-Driven RL for Diverse Data Collection
For discovering diverse trajectories, we leverage curiosity feedback for enabling the learning of perception networks to reversely affect the learning of RL policy.

Specifically, after pretraining the RL policy and training the Trajectory Scoring Module with RL-collected data,
we finetune the two parts jointly with curiosity-feedback enabled.

    sh scripts/run_train_Curiosity_RL_PushDoor.sh

### Collect Data using Trained Curiosity-Driven RL
As the policy of the curiosity-driven RL differs during different epochs, 
we collect diverse action trajectories data using policies at different epochs.


Please modify `scripts/run_collect_PushDoor.sh` to collect data using different checkpoints (we use epoch 0, 500, 1000, 1500, 2000, 2500, 3000 in our paper) of the trained curiosity-driven RL.

### Train All Perception Models
First, train the Trajectory Scoring Module and the Trajectory Proposal Module using data collected by curiosity-driven RL.

These two modules respectively rate the success likelihood of the input action, and propose diverse actions that can complete the input task.

    sh scripts/run_train_critic_PushDoor.sh
    sh scripts/run_train_actor_PushDoor.sh

Then, train the Actionability Prediction Module using the trained Trajectory Scoring Module and Trajectory Proposal Module.

This module predicts a per-point actionability score over the input shape.

    sh scripts/run_train_score_PushDoor.sh


## Evaluation and Visualization

To evaluate and visualize the results, run the following command:

```
sh scripts/run_eval_sampleSucc_PushDoor.sh
```

This will use the pretrained networks to propose interactions in the simulation, and the manipulation results are saved in `gif` format. 

