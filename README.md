# A Deep Clustering Model with Pseudo Label Correction and Distribution Alignment for Image Clustering(PCDAC)
![pcdac](https://user-images.githubusercontent.com/113669330/190835500-70aa3f9a-ff84-4056-9a8f-c656bcf28a1b.jpg)

By Feng Zhang, Zi-Wei Liu, Chee Peng Lim, Chun-Ru Dong, and Qiang Hua

This is a Pytorch implementation of the paper.

## Installation
You need to meet the following versions
```
python>=3.7
pytorch>=1.6.0
torchvision>=0.8.1
munkres>=1.1.4
numpy>=1.19.2
opencv-python>=4.4.0.46
pyyaml>=5.3.1
scikit-learn>=0.23.2
cudatoolkit>=11.0
```
Then, clone this repo
```
git clone https://github.com/LiuZiweiAI/PCDAC.git
cd PCDAC
```

## Configuration
There is a configuration file "config/config.yaml", where one can edit both the training and test options.


## Training
After setting the configuration, to start training, simply run
```
pyrhon train.py
```
If you want to train STL-10, simply run
```
python train_STL10.py
```
## Test
Once the training is completed, there will be a saved model in the "model_path" specified in the configuration file. To test the trained model, run
```
python cluster.py
```
We uploaded the pretrained model which achieves the performance reported in the paper to the Google Cloud Disk and Baidu Online Disk for reference.
```
Google Cloud：https://drive.google.com/drive/folders/1Hc9DALfyjJF1ycBDjx0pgdS6cpW2gWXw?usp=sharing
```

```
Baidu Online Disk：
  link：https://pan.baidu.com/s/1nEBd_gjzHnijC65M_d6nVQ 
  Extraction code：1234
```
