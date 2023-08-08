# UHRNet：Digital image-correlation for 2D displacement measurement Based on Unsupervised Neural Network
 project implementation (pytorch)
## Introduction
In this paper,We propose the establishment of an unsupervised framework for training the Digital image-correlation Neural Network for two-dimensional displacement measurement. To our knowledge, this is the first time that unsupervised learning has been applied to Digital Image-Correlation for displacement measurement.The predicted and original reference speckle images are compared to achieve unsupervised training. Our proposed method eliminates the need for extensive training dataannotation.The large displacement and small displacement datasets were selected to test the model and the model demonstrate its validity and robustness.


**Frame of UHRNet**
 
- Unsupervised learning structure
![Unsupervised learning structure])
- DICNet-corr structure
![DICNet-corr]()

## Main Results
-   **Prediction evalution of DICNet-corr with supervised learning and unsupervised learning

### Large displacement field

|Displacement field|Model|RMSE(mm)|SSIM|Param(M)|Speed(s)|
|---|---|---|---|---|---|
|u|DICNet-corr with supervise learning|0.433|0.9978|30.33|0.0224|
|---|DICNet-corr with unsupervise learning|1.330|0.9767|8.63|0.0093|
|v|DICNet-corr with supervise learning|0.433|0.9978|30.33|0.0224|
|---|DICNet-corr with unsupervise learning|1.330|0.9767|8.63|0.0093|
|total|DICNet-corr with supervise learning|1.330|0.9767|8.63|0.0093|
|---|DICNet-corr with unsupervise learning|1.330|0.9767|8.63|0.0093|

### small displacement field

-   **3D height map reconstructed by our method**

1. 

![demo]()

2. 

![demo]()

3.

![demo]()

4.

![demo]()

## Our Environment

- Python 3.9.7
- pytorch 1.5.0
- CUDA 11.3
- Numpy 1.23.3
## Pretrained model and Dataset
- Pretrained model(supervised learning):

- Pretrained model(unsupervised learning):
- 
According the link given above to download the weights to the UHRNet folder to run the pre-trained model
- Dataset:
## Citation

