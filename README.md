# UEO
## _Uncertainty Estimation Optimization for Reliable Remote Sensing Single-Image Super-Resolution_

## Environment
- python 3.8
- pytorch=2.1.0

## Model
![Structure of UEO](https://github.com/yaoxudong241/UEO/blob/main/image/method_00.png)
Overall architecture of the proposed uncertainty-aware SR framework. (a) An uncertainty estimation branch is integrated into a conventional SR network, enhanced with a sparse sampling module to promote sparsity in uncertainty representation. The output SR image $I_{SR}$ and the corresponding uncertainty map $U$ are jointly used to compute the uncertainty-aware loss function $L_U$. (b) A VGG-based perceptual similarity constraint $L_D$ between SR and HR images is introduced to regularize and stabilize the uncertainty learning process. $L_U$ and $L_D$ jointly constitute the overall optimization objective of the proposed framework.

## Train
- dataset:AID
- prepare

```sh
python .\main_fat_step1_weight.py
```
```sh
python .\main_fat_step2_weight.py
```
## Test

```sh
python .\main_fat_step2_weight_test.py
```


## Visual comparison
![playground](https://github.com/user-attachments/assets/68204de0-f7db-4919-95a9-c10047b5c5ce)
![road](https://github.com/user-attachments/assets/c1183624-16cd-4eda-aeb3-a711048c93b9)
![ship](https://github.com/user-attachments/assets/1d434a3a-3998-4074-8312-29736034bcb7)
Qualitative evaluation results for different SISR methods on AID dataset at a scale of 4. (a), (b) and (c) correspond to the names of images in the dataset. Our results restore sharper and more accurate boundaries, which are closer to the ground truth.

## Results
The super-resolution result images for AID, UCMerced, and SEN2VENµS can be obtained from [Google Drive](https://drive.google.com/drive/folders/17Vyd9NSD6gFQk5OjHMEgd3dhlvY8a3V4?usp=drive_link).

## Acknowledgements
This code is built on [ESRT (Torch)](https://github.com/luissen/ESRT). We thank the authors for sharing their codes of ESRT PyTorch version.

