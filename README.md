## [Efficient Slice Anomaly Detection Network for 3D brain MRI Volume]

PyTorch implementation for paper, Efficient Slice Anomaly Detection Network for 3D brain MRI Volume.

## Download Datasets

Please download MVtecAD dataset from [MVTecAD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/), Brats2021 dataset from [BraTS 2021 dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1), Br35H dataset from [Br35H dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) and IXI dataset from [IXI dataset](https://brain-development.org/ixi-dataset/).

## Training
- Run code for 2D training MVTecAD with SimpleNet
```
python train_2D.py --score_model disc --dataset MVTec --class_name screw --proj_layers 1 --preprocessing_dimension 1536 --target_embed_dimension 1536 --noise_std 0.015
```

- Run code for 2D training MVTecAD with our model SimpleSliceNet
```
python train_2D.py --score_model flow --dataset MVTec --proj_layers 0 --preprocessing_dimension 512 --target_embed_dimension 512 --noise_std 0.08
```

- Run code for 2D training BraTS2021 with SimpleNet
```
python train_2D.py --score_model disc --dataset BraTS --proj_layers 1 --preprocessing_dimension 1536 --target_embed_dimension 1536 --noise_std 0.08
```

- Run code for 2D training BraTS2021 with our model SimpleSliceNet
```
python train_2D.py --score_model flow --dataset BraTS --project_layer 0 --preprocessing_dimension 512 --target_embed_dimension 512 --noise_std 0.08
```

- Run code for 3D training BraTS2021 with our model SimpleSliceNet
```
python train_3D.py
```

- Using feature extractor fine-tuned on EDC.
Please use the code from original paper [EDC](https://github.com/guojiajeremy/edc), get the state_dict of the encoder and save it as './../results/MRI_EDC/best_encoder.pth'.
 ## Pre-processing
 - [src/data_utils.py](src/data_utils.py) contains code for brain extraction, volume registered and converting volumes to slices. As mentioned in paper, we use [HD-BET](https://github.com/MIC-DKFZ/HD-BET) to skull-tripped and use [Flirt](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) to registered brain MRI volumes.
