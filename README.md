# Caffe TVG Segmentation

This fork of [Caffe](https://github.com/BVLC/caffe) supports the various projects on semantic-, instance- and panoptic-segmentation done by the [Torr Vision Group](http://www.robots.ox.ac.uk/~tvg/).

These include:

- [Conditional Random Fields as Recurrent Neural Networks](https://github.com/torrvision/crfasrnn), ICCV 2015.

- [Higher Order Conditional Random Fields in Deep Neural Networks](http://www.robots.ox.ac.uk/~aarnab/projects/eccv_2016/Higher_Order_CRF_CNN.pdf), ECCV 2016.

- [Bottom-up Instance Segmentation with Deep Higher-Order CRFs](http://www.robots.ox.ac.uk/~aarnab/projects/bmvc_2016/InstanceSegmentation.pdf), BMVC 2016.

- [Pixelwise Instance Segmentation with a Dynamically Instantiated Network](https://github.com/hmph/dynamically-instantiated-network), CVPR 2017.

- [Holistic, Instance-Level Human Parsing](http://www.robots.ox.ac.uk/~aarnab/projects/bmvc_2016/InstanceSegmentation.pdf), BMVC 2017.

- [On the Robustness of Semantic Segmentation Models to Adversarial Attacks](http://www.robots.ox.ac.uk/~aarnab/adversarial_robustness.html), CVPR 2018, PAMI 2020.

- [Weakly- and Semi-Supervised Panoptic Segmentation](https://github.com/qizhuli/Weakly-Supervised-Panoptic-Segmentation). ECCV 2018.

Note that this code base has been refactored and modified since the original papers were written.

## Installation

Follow the standard guide for installing [Caffe](README_caffe.md).

## Credits

In addition to the original [Caffe](https://github.com/BVLC/caffe), this repository contains layers from [Fast-RCNN](https://github.com/rbgirshick/caffe-fast-rcnn/tree/bcd9b4eadc7d8fbc433aeefd564e82ec63aaf69c), [R-FCN](https://github.com/daijifeng001/caffe-rfcn/tree/4bcfcd104bb0b9f0862e127c71bd845ddf036f14), [Dilated Convolutions](https://github.com/fyu/caffe), [Deeplab](https://bitbucket.org/aquariusjay/deeplab-public-ver2/src/master/), [PSPNet](https://github.com/hszhao/PSPNet), [MNC](https://github.com/daijifeng001/caffe-mnc/tree/d8bf82b7dae8e48e098d3316860c4a86f847b1e6) and [Intel Caffe](https://github.com/intel/caffe).

## Contact

For any queries, please contact Anurag Arnab (aarnab@robots.ox.ac.uk). Pull requests are also welcome.