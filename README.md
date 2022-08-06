# VAT-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D ARTiculated Objects



This repository provides source code for our paper:

[VAT-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D ARTiculated Objects](https://hyperplane-lab.github.io/vat-mart/)

[Ruihai Wu](https://warshallrho.github.io)\*, [Yan Zhao](https://sxy7147.github.io)\*, [Kaichun Mo](https://www.cs.stanford.edu/~kaichun)\*, [Zizheng Guo](https://guozz.cn/), [Yian Wang](https://galaxy-qazzz.github.io/), [Tianhao Wu](https://tianhaowuhz.github.io/), [Qingnan Fan](https://fqnchina.github.io/), [Xuelin Chen](https://xuelin-chen.github.io/), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/), [Hao Dong](https://zsdonghao.github.io/) (\* indicates joint first authors)

ICLR, 2022

Currently we have released the code for perception networks, and we are working on purifying and releasing other code in recent few days (08/06/2022).

Project Page: https://hyperplane-lab.github.io/vat-mart/


## Dependencies

This code has been tested on Ubuntu 18.04 with Cuda 10.1, Python 3.6, and PyTorch 1.7.0.

First, install SAPIEN following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl

For other Python versions, you can use one of the following

    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp35-cp35m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
    pip install http://download.cs.stanford.edu/orion/where2act/where2act_sapien_wheels/sapien-0.8.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

Please do not use the default `pip install sapien` as SAPIEN is still being actively developed and updated.

Then install PointNet++ as we need to process the point cloud.

    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    # [IMPORTANT] comment these two lines of code:
    #   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
    pip install -r requirements.txt
    pip install -e .

The other requirements could be installed directly by pip install.

For visualization, please install blender v2.79 and put the executable in your environment path.
Also, the prediction result can be visualized using MeshLab or the *RenderShape* tool in [Thea](https://github.com/sidch/thea).



## Citations

Please cite our work if you find it useful:

    @inproceedings{
    wu2022vatmart,
    title={{VAT}-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D {ART}iculated Objects},
    author={Ruihai Wu and Yan Zhao and Kaichun Mo and Zizheng Guo and Yian Wang and Tianhao Wu and Qingnan Fan and Xuelin Chen and Leonidas Guibas and Hao Dong},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=iEx3PiooLy}
    }


## License

MIT Licence
