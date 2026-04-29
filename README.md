# FD-SCU

This is the official implementation of **[FD-SCU: Frequency Decomposition-based
Spectrum Collaborative Upsampling for Point Cloud Color Attribute](https://ieeexplore.ieee.org/document/11466445/)**.

## Prerequisite

Install prerequisites using the following command. 
Note that we are using PyTorch 1.11.0 + CUDA 11.3. 
Please adjust the version according to your environment.

```
pip install -r requirements.txt
```

## Datasets
### MPEG 8i
Download the MPEG 8i dataset from:
https://mpeg-pcc.org/index.php/pcc-content-database/8i-voxelized-surface-light-field-8ivslf-dataset/

### WPC
Download the WPC dataset from:
https://github.com/qdushl/Waterloo-Point-Cloud-Database
### FaceScape

Apply for and download the FaceScape dataset from:
https://facescape.nju.edu.cn/

Organize the dataset as follows. Data splitting files (train.txt, valid.txt, test.txt) can be found in data/FaceScape/.


```
FaceScape\
    origin\
        TU-Model\
            1\
            2\
            ...
    train.txt
    valid.txt
    test.txt
```

Run the following command to generate a point cloud dataset from the orignal FaceScape dataset.

```
cd data/
python face_scape_preprocess.py --data_dir XXX/FaceScape/
```

## Test


Pretrained FD-SCU models can be found in the pretrained/ folder.

Run the following file to test FD-SCU and other baselines on the FaceScape dataset:

```
python test.py
```


## Train

Run the following file to train FD-SCU:

```
python train.py
```


## Citation

If FD-SCU is useful for your research, please consider citing:

```

@ARTICLE{11466445,
  author={Liu, Hao and Wang, Wenchao and Yuan, Hui and Hamzaoui, Raouf and Yan, Weiqing and Hou, Junhui},
  journal={IEEE Transactions on Image Processing}, 
  title={FD-SCU: Frequency Decomposition-Based Spectrum Collaborative Upsampling for Point Cloud Color Attribute}, 
  year={2026},
  volume={35},
  number={},
  pages={3522-3536},
  keywords={Feeds;Antennas;Broadcasting;Circuits and systems;Filtering;Filters;Videos;Video equipment;High frequency;Radio frequency;Color upsampling;virtual filling;frequency decomposition;Gaussian perturbation},
  doi={10.1109/TIP.2026.3678090}}


```

## Questions
Please contact 'wenchaowangx@gmail.com'
