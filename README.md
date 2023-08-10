# Digital image-correlation for 2D displacement measurement Based on Unsupervised Neural Network
 project implementation (pytorch)
## Introduction
In this paper,We propose the establishment of an unsupervised framework for training the Digital image-correlation Neural Network for two-dimensional displacement measurement. To our knowledge, this is the first time that unsupervised learning has been applied to Digital Image-Correlation for displacement measurement.The predicted and original reference speckle images are compared to achieve unsupervised training. Our proposed method eliminates the need for extensive training dataannotation.The large displacement and small displacement datasets were selected to test the model and the model demonstrate its validity and robustness.


**Frame of UHRNet**
 
- Unsupervised learning structure
![Unsupervised learning structure](https://github.com/fead1/DICNet-corr-unsupervised-learning-/blob/main/unsuperivise%20learning%20with%20DICNet-coor/Net%20Structure/Unsupervised%20learning%20structure.png)
- DICNet-corr structure
![DICNet-corr](https://github.com/fead1/DICNet-corr-unsupervised-learning-/blob/main/unsuperivise%20learning%20with%20DICNet-coor/Net%20Structure/DICNet-corr%20structure.png)

## Main Results
-   **Prediction evalution of DICNet-corr with supervised learning and unsupervised learning

### Large displacement field

|Displacement field|Model|Mean MAE(pixel)|Mean RMSE(pixel)|
|---|---|---|---|
|u|DICNet-corr with supervise learning|0.0588|0.0759|
|---|DICNet-corr with unsupervise learning|0.0639|0.0834|
|v|DICNet-corr with supervise learning|0.0665|0.0850|
|---|DICNet-corr with unsupervise learning|0.0723|0.0927|
|total|DICNet-corr with supervise learning|0.0626|0.0807|
|---|DICNet-corr with unsupervise learning|0.0681|0.0886|

### small displacement field

|Displacement field|Model|Mean MAE(pixel)|Mean RMSE(pixel)|
|---|---|---|---|
|u|DICNet-corr with supervise learning|0.0221|0.0299|
|---|DICNet-corr with unsupervise learning|0.0309|0.0453|
|v|DICNet-corr with supervise learning|0.0209|0.00285|
|---|DICNet-corr with unsupervise learning|0.0308|0.0452|

-   **3D height map reconstructed by our method**

1. 

![large displacement field predict](https://github.com/fead1/DICNet-corr-unsupervised-learning-/blob/main/unsuperivise%20learning%20with%20DICNet-coor/Image%20show/large_dis.png)

2. 

![small displacement field predict](https://github.com/fead1/DICNet-corr-unsupervised-learning-/blob/main/unsuperivise%20learning%20with%20DICNet-coor/Image%20show/small_dis.png)

3.

![Star5 displacement field predict](https://github.com/fead1/DICNet-corr-unsupervised-learning-/blob/main/unsuperivise%20learning%20with%20DICNet-coor/Image%20show/Star5.png)


## Our Environment

- Python 3.9.7
- pytorch 1.5.0
- CUDA 11.3
- Numpy 1.23.3
## Pretrained model and Dataset
- Pretrained model(with large displacement dataset):
Link：https://pan.baidu.com/s/1mkZhYRCNXlnXrVForYECSQ 
Password：18qf
- Pretrained model(with small displacement dataset):
Link：https://pan.baidu.com/s/1bhTCOSXd119DtZOuUVOlkw 
Password：hcm4
According the link given above to download the weights to the UHRNet folder to run the pre-trained model
- Dataset:
## Citation

