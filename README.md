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
![results](https://github.com/yaoxudong241/UEO/blob/main/image/results.png)
The qualitative evaluation results of different methods on the AID dataset with a scale factor of 4. The scene category and image name are annotated in the titles of subfigures (a)–(c), and these three images correspond to the SR results are generated using the HAT, MEN, and MSCT models, respectively. In each subfigure, the second row presents a magnified view of the red-boxed region from the first row.

## Results
Taking the FAT network as an example, our experimental results and pretrained weights are available at [Google Drive](https://drive.google.com/drive/folders/1e431aeaxCxif6ggIX2Ebe9DGkTJPFNz7?usp=drive_link).

## Acknowledgements
This code is built on [Uncertainty-Driven Loss for Single Image Super-Resolution (Torch)](https://github.com/QianNing0/UDL). We thank the authors for sharing their codes of ESRT PyTorch version.

