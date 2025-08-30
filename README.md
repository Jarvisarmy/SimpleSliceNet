
## Download Datasets

Please download MVtecAD dataset from [MVTecAD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/), Brats2021 dataset from [BraTS 2021 dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1), Br35H dataset from [Br35H dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) and IXI dataset from [IXI dataset](https://brain-development.org/ixi-dataset/).

## Files and Tools Setup
- Using feature extractor fine-tuned on EDC.
 Please use the code from original paper [EDC](https://github.com/guojiajeremy/edc), get the state_dict of the encoder and save it as './../results/MRI_EDC/best_encoder.pth'. You can also contact us for inquiring our saved state dict.

- As mentioned in paper, we use [HD-BET](https://github.com/MIC-DKFZ/HD-BET) to skull-tripped and use [Flirt](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) to registered brain MRI volumes. Please download the MRI preprocessing tool follow the description from the source website.  These tools are download in the parent directory of this repository, if you download it elsewhere, you might need to change some code in the [src/data_utils.py](src/data_utils.py) to make sure the pre-processing code works.

## Pre-processing
 - [src/data_utils.py](src/data_utils.py) contains code for brain extraction, volume registered and converting volumes to slices. 
 - function [brain_extraction] is used to perform brain extraction using HD-BET
 - function [registered_nii_IXIT2] is used to registered the volumes from IXI to standard template
 - function [registered_nii_BraTS] is used to registered the volumes from BraTS to standard template
 - function [load_slices_for_3D] is used convert each volume to slices and stored in folder BraTS2D which is used to trained 2D methods.
You can directly run the script below to preprocess the data. Just make sure that under your data root directory, the IXI T1 volumes are saved in a folder named "IXI-T1", the IXI T2 volumes in "IXI-T2", and the BraTS2021 data in "BraTS2021". The script will automatically create two folders, "T1" and "T2", under the root path and save all preprocessed data into them.
```
python data_utils.py --data_base_dir \path\to\data\root
```

## Training
- Run code for 2D training MVTecAD with SimpleNet
```
python train_2D.py --score_model disc --dataset MVTec --class_name screw --proj_layers 1 --preprocessing_dimension 1536 --target_embed_dimension 1536 --noise_std 0.015 --use_gpu
```

- Run code for 2D training MVTecAD with our model SimpleSliceNet
```
python train_2D.py --score_model flow --dataset MVTec --proj_layers 0 --preprocessing_dimension 512 --target_embed_dimension 512 --noise_std 0.08 --use_gpu
```

- Run code for 2D training BraTS2021 with SimpleNet
```
python train_2D.py --score_model disc --dataset BraTS --proj_layers 1 --preprocessing_dimension 1536 --target_embed_dimension 1536 --noise_std 0.08 --use_gpu
```

- Run code for 2D training BraTS2021 with our model SimpleSliceNet
```
python train_2D.py --score_model flow --dataset BraTS --proj_layer 0 --preprocessing_dimension 512 --target_embed_dimension 512 --noise_std 0.08 --edc --use_gpu
```

- Run code for 3D training BraTS2021 with our model SimpleSliceNet
```
python train_3D.py --score_model flow --dataset BraTS --proj_layer 0 --preprocessing_dimension 512 --target_embed_dimension 512 --noise_std 0.08 --edc --use_gpu
```


