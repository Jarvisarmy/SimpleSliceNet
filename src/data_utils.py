"""
This file contains all the necessary code to preprocess the t2 weighted volume from IXI and BraTS2021
link to IXI: http://brain-development.org/ixi-dataset/
link to BraTS: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
"""


from PIL import Image
import os
import nibabel as nib
import numpy as np
import torch
import argparse

from nii_utils import crop_3d_volume_to_size

def load_slices_from_BraTS(data_root):
    """
    1000 training patiens, 251 test patients
    """
    subjects = []
    
    input_path = os.path.join(data_root, "BraTS2021")
    
    for file in os.listdir(input_path):
        if file.startswith('BraTS'):
            subjects.append(file)
    train_subjects = subjects[:1000]
    test_subjects = subjects[1000:]
    '''
    os.mkdir(os.path.join(data_root,'BraTS2D'))
    for dir in ['train','val','test']:
        os.mkdir(os.path.join(data_root,'BraTS2D',dir))
        os.mkdir(os.path.join(data_root,'BraTS2D',dir,'images'))
        os.mkdir(os.path.join(data_root,'BraTS2D',dir,'masks'))
    '''
    train_data = os.path.join(data_root, "BraTS2D/train/images")
    train_mask = os.path.join(data_root, 'BraTS2D/train/masks')
    val_data = os.path.join(data_root, "BraTS2D/val/images")
    val_mask = os.path.join(data_root, "BraTS2D/val/masks")
    test_data =  os.path.join(data_root, "BraTS2D/test/images")
    test_mask =  os.path.join(data_root, "BraTS2D/test/masks")
    
    # for each subjects in training, add normal slices to training and abnormal slices to validation
    train_normal= 0
    train_abnormal=0
    min_idx =100
    max_idx = 0
    def store_image_from_array(image_folder,mask_folder,subject,image,mask,slice_idx):
        slice_image = Image.fromarray(image, mode='L')
        slice_mask = Image.fromarray(mask, mode='L')
        png_filename = os.path.join(image_folder, '{}_{}{}'.format(subject,slice_idx,".png"))
        mask_filename = os.path.join(mask_folder, '{}_{}{}'.format(subject,slice_idx,".png"))
        slice_image.save(png_filename)
        slice_mask.save(mask_filename)
    for subject in train_subjects:
        volume_path = os.path.join(input_path, subject, subject+"_t2.nii.gz")
        mask_path = os.path.join(input_path, subject, subject+"_seg.nii.gz")
        volume = nib.load(volume_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        volume = crop_3d_volume_to_size(volume, target_size=(192,192,20))
        if np.ptp(volume) == 0:
            volume = np.zeros_like(volume, dtype='uint8')
        else:
            volume = ((volume-volume.min())/(volume.max()-volume.min())*255).astype('uint8')
        mask = crop_3d_volume_to_size(mask,target_size=(192,192,20))
        for slice_idx in range(volume.shape[2]):
            slice = volume[:,:,slice_idx]
            
            slice_mask = mask[:,:,slice_idx].astype('uint8')
            if np.max(slice_mask) > 0:
                train_abnormal += 1
                #store_image_from_array(val_data,val_mask,subject, slice, slice_mask,slice_idx)
            else:
                train_normal += 1
                #store_image_from_array(train_data,train_mask,subject,slice,slice_mask,slice_idx)
    test_normal = 0
    test_abnormal = 0
    for subject in test_subjects:
        volume_path = os.path.join(input_path, subject, subject+"_t2.nii.gz")
        mask_path = os.path.join(input_path, subject, subject+"_seg.nii.gz")
        volume = nib.load(volume_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        volume = crop_3d_volume_to_size(volume, target_size=(192,192,20))
        if np.ptp(volume) == 0:
            volume = np.zeros_like(volume, dtype='uint8')
        else:
            volume = ((volume-volume.min())/(volume.max()-volume.min())*255).astype('uint8')
        mask = crop_3d_volume_to_size(mask,target_size=(192,192,20))
        for slice_idx in range(volume.shape[2]):
            slice = volume[:,:,slice_idx]
        
            slice_mask = mask[:,:,slice_idx].astype('uint8')
            if np.max(slice_mask) > 0:
                test_abnormal += 1
                #store_image_from_array(test_data,test_mask,subject, slice, slice_mask,slice_idx)
            else:
                test_normal += 1
                #store_image_from_array(test_data,test_mask,subject, slice, slice_mask,slice_idx)
    print('train_normal', train_normal)
    print('train_abnormal', train_abnormal)
    print('test_normal', test_normal)
    print('test_abnormal', test_abnormal)

        
            

def convert_volume_to_slices(data_root,resize_size=None,target_size=None,slices_idx=None):
    input_IXI = os.path.join(data_root, "IXI_registered")
    input_BraTS = os.path.join(data_root, 'BraTS_registered')
    input_BraTS_mask = os.path.join(data_root, 'BraTS_mask_registered')

    output_IXI = os.path.join(data_root, "IXI_middle")
    output_BraTS = os.path.join(data_root, "BraTS_middle")
    output_BraTS_mask = os.path.join(data_root, "BraTS_mask_middle")
    os.mkdir(output_IXI)
    os.mkdir(output_BraTS)
    os.mkdir(output_BraTS_mask)
    def _extract(input_path,output_path):
        for file in os.listdir(input_path):
            if file.startswith('IXI') or file.startswith("BraTS"):
                volume_path = os.path.join(input_path, file)
                volume = nib.load(volume_path).get_fdata()
                if target_size:
                    volume = crop_3d_volume_to_size(volume, resize_size,target_size)
                if slices_idx:
                    slice_choices =slices_idx
                else:
                    slice_choices = range(volume.shape[2])
                for slice_idx in slice_choices:
                    slice_2d = volume[:,:,slice_idx]
                    if np.ptp(slice_2d) == 0:
                        slice_2d = np.zeros_like(slice_2d, dtype='uint8')
                    else:
                        slice_2d = ((slice_2d-slice_2d.min())/(slice_2d.max()-slice_2d.min())*255).astype('uint8')
                    slice_image = Image.fromarray(slice_2d, mode='L')
                    sub_dirs = file.split(".")
                    png_filename = os.path.join(output_path, '{}_{}{}'.format(sub_dirs[0],slice_idx,".png"))
                    slice_image.save(png_filename)
    _extract(input_IXI, output_IXI)
    _extract(input_BraTS, output_BraTS)
    _extract(input_BraTS_mask, output_BraTS_mask)

def brain_extraction(in_path, out_path):
    """use HD-BET to perform brain extraction
        use following comand in terminal to download HD-BET
            #!git clone https://github.com/MIC-DKFZ/HD-BET
            #!pip install -e ./HD-BET
            #!mkdir ./IXIT2_skull-stripped
    """
    if torch.cuda.is_available():
        hd_bet_extraction = "hd-bet -i {} -o {}".format(in_path, out_path)
    else:
        hd_bet_extraction = "hd-bet -i {} -o {} -device cpu -mode fast -tta 0"
    os.system(hd_bet_extraction)

def registered_nii_IXI(base_template, data_root,input_folder):
    IXI_registered_path = os.path.join(data_root, 'IXI_registered')
    input_path = os.path.join(data_root, input_folder)
    if not os.path.exists(IXI_registered_path):
        os.makedirs(IXI_registered_path)
    for file in os.listdir(input_path):
        if file.endswith('T2.nii.gz'):
            in_file_path = os.path.join(input_path, file)
            out_file_path = os.path.join(IXI_registered_path, file)
            data_registered_command = "flirt -in {} -ref {}  -applyisoxfm 1.0 -out {}".format(in_file_path, base_template, out_file_path)
            os.system(data_registered_command)
def registered_nii_BraTS(base_template, data_root, data_folder, mask_folder):
    # base_template can be ./fsl/data/standard/MNI152_T1_1mm.nii.gz for 1mm criterion

    BraTS_data_path = os.path.join(data_root, data_folder)
    BraTS_mask_path = os.path.join(data_root, mask_folder)
    BraTS_transformation_path = os.path.join(data_root,'BraTS_transformation')
    BraTS_registered_path = os.path.join(data_root, 'BraTS_registered')
    BraTS_mask_registered_path = os.path.join(data_root, 'BraTS_mask_registered')
    BraTS_reversed_path = os.path.join(data_root,'BraTS_reversed')
    BraTS_mask_reversed_path = os.path.join(data_root,'BraTS_mask_reversed')
    if not os.path.exists(BraTS_transformation_path):
        os.makedirs(BraTS_transformation_path)
    if not os.path.exists(BraTS_registered_path):
        os.makedirs(BraTS_registered_path)
    if not os.path.exists(BraTS_mask_registered_path):
        os.makedirs(BraTS_mask_registered_path)
    if not os.path.exists(BraTS_reversed_path):
        os.makedirs(BraTS_reversed_path)
    if not os.path.exists(BraTS_mask_reversed_path):
        os.makedirs(BraTS_mask_reversed_path)
    for file in os.listdir(BraTS_data_path):
        if file.endswith('t2.nii.gz'):
            in_file_path = os.path.join(BraTS_data_path, file)
            reversed_file_path = os.path.join(BraTS_reversed_path, 'reversed_'+file)
            out_file_path = os.path.join(BraTS_registered_path, file)
            in_mask_path = os.path.join(BraTS_mask_path, file.replace('t2.nii.gz','seg.nii.gz'))
            reversed_mask_path = os.path.join(BraTS_mask_reversed_path, 'reversed_'+file.replace('t2.nii.gz','seg.nii.gz'))
            out_mask_path = os.path.join(BraTS_mask_registered_path, file.replace('t2.nii.gz','seg.nii.gz'))
            out_matrix_path = os.path.join(BraTS_transformation_path, file.replace('nii.gz','mat'))
            data_reversed_command = "fslswapdim {} x -y z {} > /dev/null 2>&1".format(in_file_path, reversed_file_path)
            mask_reversed_command = "fslswapdim {} x -y z {} > /dev/null 2>&1".format(in_mask_path, reversed_mask_path)
            data_registered_command= "flirt -in {} -ref {} -out {} -omat {} -applyisoxfm 1.0 2> /dev/null".format(reversed_file_path, base_template, out_file_path, out_matrix_path)
            mask_registered_command = "flirt -in {} -ref {} -out {} -init {} -applyisoxfm 1.0 2> /dev/null".format(reversed_mask_path, base_template, out_mask_path, out_matrix_path)
            os.system(data_reversed_command)
            os.system(mask_reversed_command)
            os.system(data_registered_command)
            os.system(mask_registered_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output directories.")
    parser.add_argument('--data_base_dir', type=str, required=True, help="Path to the data base directory")
    args = parser.parse_args()
    IXIT1_path = os.path.join(args.data_base_dir, 'IXI-T1')
    IXIT2_path = os.path.join(args.data_base_dir, 'IXIT2')
    BraTS_path = os.path.join(args.data_base_dir, 'BraTS2021')
    # step 1: perform brain extraction: skull-stripped
    print(('-------------------- brain extraction on IXIT1 --------------------'))
    brain_extraction(IXIT1_path, )


    print(('-------------------- brain extraction on IXIT1 --------------------'))
    
    
    load_slices_from_BraTS("./../data")
