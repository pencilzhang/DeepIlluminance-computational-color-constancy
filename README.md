﻿# DeepIlluminance: Contextual Illuminance Estimation via Deep Neural Networks

We release the Caffe code of [DeepIlluminance](https://arxiv.org/abs/1905.04791).


### Reference

If you find our paper and repo useful, please cite our paper. Thanks!

```
@article{Zhang2019illuminance,
    title={DeepIlluminance: Contextual Illuminance Estimation via Deep Neural Networks},
    author={Zhang, Jun and Zheng, Tong and Zhang, Shengping and Wang, Meng},
    journal={arXiv preprint arXiv:1905.04791},
    year={2019}
}  
```

### Prerequisites

The code is built with following libraries:

- [Caffe](https://caffe.berkeleyvision.org/)
- [Python 2.7](https://www.anaconda.com/distribution/)
- [Matlab 2016b](https://www.mathworks.com/products/matlab.html)

### Data pre-processing

- We have trained on ColorChecker and NUS-8 datasets. Please refer to [ColorChecker](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) and [NUS-8](http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html) datasets for the detailed guide of data generation.
Basically, the processing of image data can be summarized into 3 steps:
- Sample image patches containing both bright and dark pixels (refer to [search_patch_neighbor.py](./search_patch_neighbor.py)) 
- View the gamma correction patches (refer to [gamma.m](./gamma.m))
- Generate LMDB files (refer to [create_data_lmdb.sh](./create_data_lmdb.sh) and [create_lmdb.py](./create_lmdb.py))


### ColorChecker pretrained models

Here we provide the pretrained models on ColorChecker for fine-tuning at WeYun: https://share.weiyun.com/50GG5jx or GoogleDrive: https://drive.google.com/open?id=15tvz2DzlCQi3VgOpghGtkCwcf1giorWE.



### Testing 

For example, to test the downloaded pretrained models on ColorChecker, you can run `python context_network/trained_model/test.py` to get the output of the contextual network. Then, run `python refinement_network/trained_model/test.py` to get the final estimation result.


### Training 

We provide codes to train DeepIlluminance network with this repo:
  For the contextual network: run `python ./context_network/train.py`
  For the refinement network: run `python ./refinement_network/train.py`
