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
import shutil
import torch.nn.functional as F
def crop_3d_volume_to_size(data,resize_size=None, target_size=None):

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif isinstance(data, list):
        data = torch.tensor(data)
    
    h, w, d = data.shape
    if target_size:
        crop_h, crop_w, crop_d= target_size

        crop_d_start = max(0, (d - crop_d) // 2)
        crop_h_start = max(0, (h - crop_h) // 2)
        crop_w_start = max(0, (w - crop_w) // 2)

        crop_d_end = min(d, crop_d_start + crop_d)
        crop_h_end = min(h, crop_h_start + crop_h)
        crop_w_end = min(w, crop_w_start + crop_w)
        
        #crop_h_end = h-1
        #crop_h_start = crop_h_end - crop_h

        # Perform center crop
        cropped_volume = data[crop_h_start:crop_h_end, crop_w_start:crop_w_end,crop_d_start:crop_d_end]
    else:
        cropped_volume = data
    if resize_size:
        resized_volume = F.interpolate(cropped_volume.unsqueeze(0).unsqueeze(0), 
                                   size = resize_size,
                                   mode = 'trilinear',
                                   align_corners=False).squeeze(0).squeeze(0)
    else:
        resized_volume = cropped_volume
    #cropped_volume = cropped_volume.permute(2,0,1)
    #cropped_volume = F.interpolate(cropped_volume.unsqueeze(0), 
    #                               size = (224,224),
    #                               mode = 'bilinear',
    #                               align_corners=False).squeeze(0)
    #cropped_volume = cropped_volume.permute(1,2,0)
    return resized_volume.numpy()


        
            
def brain_extraction(in_path, out_path):
    """use HD-BET to perform brain extraction
        use following comand in terminal to download HD-BET
            #!git clone https://github.com/MIC-DKFZ/HD-BET
            #!pip install -e ./HD-BET
            #!mkdir ./IXIT2_skull-stripped
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if torch.cuda.is_available():
        hd_bet_extraction = "hd-bet -i {} -o {}".format(in_path, out_path)
    else:
        hd_bet_extraction = "hd-bet -i {} -o {} -device cpu -mode fast -tta 0"
    os.system(hd_bet_extraction)

def registered_nii_IXIT2(base_template,input_path, IXI_registered_path):
    if not os.path.exists(IXI_registered_path):
        os.makedirs(IXI_registered_path)
    for file in os.listdir(input_path):
        if file.endswith('T2.nii.gz'):
            in_file_path = os.path.join(input_path, file)
            out_file_path = os.path.join(IXI_registered_path, file)
            out_matrix_path = os.path.join(T2_temp, file.replace('nii.gz','mat'))
            data_transform_command = "flirt -in {} -ref {} -omat {} -out {}".format(in_file_path, base_template, out_matrix_path, out_file_path)
            data_registered_command = "flirt -in {} -ref {} -init {} -applyisoxfm 1.0 -out {}".format(in_file_path, base_template, out_matrix_path,out_file_path)
            os.system(data_transform_command)
            os.system(data_registered_command)
def registered_nii_IXIT1(base_template, IXI_skull_stripped_path, IXI_registered_path, IXI_reversed_path):
    if not os.path.exists(IXI_registered_path):
        os.makedirs(IXI_registered_path)
    if not os.path.exists(IXI_reversed_path):
        os.makedirs(IXI_reversed_path)
    for file in os.listdir(IXI_skull_stripped_path):
        if file.endswith('T1.nii.gz'):
            in_file_path = os.path.join(IXI_skull_stripped_path, file)
            reversed_file_path = os.path.join(IXI_reversed_path, 'reversed_'+file)
            out_file_path = os.path.join(IXI_registered_path, file)
            out_matrix_path = os.path.join(T1_temp, file.replace('nii.gz','mat'))
            data_reversed_command = "fslswapdim {} z -x y {} > /dev/null 2>&1".format(in_file_path, reversed_file_path)
            data_transform_command = "flirt -in {} -ref {} -omat {} -out {}".format(reversed_file_path, base_template, out_matrix_path, out_file_path)
            data_registered_command = "flirt -in {} -ref {} -init {} -applyisoxfm 1.0 -out {}".format(reversed_file_path,base_template, out_matrix_path,out_file_path)
            os.system(data_reversed_command)
            os.system(data_transform_command)
            os.system(data_registered_command)
    

def registered_nii_BraTS(base_template,data_root, modality):
    if modality == 'T1':
        file_end = 't1.nii.gz'
    else:
        file_end = 't2.nii.gz'
    BraTS_data_path = os.path.join(data_root,'BraTS_data')
    BraTS_mask_path = os.path.join(data_root,'BraTS_mask')
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
        if file.endswith(file_end):
            in_file_path = os.path.join(BraTS_data_path, file)
            reversed_file_path = os.path.join(BraTS_reversed_path, 'reversed_'+file)
            out_file_path = os.path.join(BraTS_registered_path, file)
            in_mask_path = os.path.join(BraTS_mask_path, file.replace(file_end,'seg.nii.gz'))
            reversed_mask_path = os.path.join(BraTS_mask_reversed_path, 'reversed_'+file.replace(file_end,'seg.nii.gz'))
            out_mask_path = os.path.join(BraTS_mask_registered_path, file.replace(file_end,'seg.nii.gz'))
            out_matrix_path = os.path.join(BraTS_transformation_path, file.replace('nii.gz','mat'))
            data_reversed_command = "fslswapdim {} x -y z {} > /dev/null 2>&1".format(in_file_path, reversed_file_path)
            mask_reversed_command = "fslswapdim {} x -y z {} > /dev/null 2>&1".format(in_mask_path, reversed_mask_path)
            data_transform_command = "flirt -in {} -ref {} -omat {} -out {}".format(reversed_file_path, base_template, out_matrix_path, out_file_path)
            data_registered_command= "flirt -in {} -ref {} -out {} -init {} -applyisoxfm 1.0 2> /dev/null".format(reversed_file_path, base_template, out_file_path, out_matrix_path)
            mask_registered_command = "flirt -in {} -ref {} -out {} -init {} -applyisoxfm 1.0 2> /dev/null".format(reversed_mask_path, base_template, out_mask_path, out_matrix_path)
            os.system(data_reversed_command)
            os.system(mask_reversed_command)
            os.system(data_transform_command)
            os.system(data_registered_command)
            os.system(mask_registered_command)



def process_BraTS(source_path, data_root, modality):
    if modality == 'T1':
        file_end = '_t1.nii.gz'
    else:
        file_end = '_t2.nii.gz'
    data_path = os.path.join(data_root, 'BraTS_data')
    mask_path = os.path.join(data_root, 'BraTS_mask')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    for file in os.listdir(source_path):
        path = os.path.join(source_path, file)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(file_end):
                    shutil.copy(os.path.join(path, file), os.path.join(data_path,file))
                if file.endswith('_seg.nii.gz'):
                    shutil.copy(os.path.join(path, file), os.path.join(mask_path,file))

def load_slices_for_3D(data_root,BraTS_path, volume_size):
    """
    251 for val, and 1000 for testing
    """
    subjects = []
    train_subjects = []
    val_subjects = []
    test_subjects = []
    BraTS_image_path = os.path.join(data_root, "BraTS_registered")
    BraTS_mask_path = os.path.join(data_root, "BraTS_mask_registered")
    IXI_path = os.path.join(data_root, "IXI_registered")
    volume_size = volume_size
    for file in os.listdir(BraTS_path):
        if file.startswith('BraTS'):
            subjects.append(file)
    for file in os.listdir(IXI_path):
        if file.startswith('IXI'):
            train_subjects.append(file)
    val_subjects = subjects[:251]
    test_subjects = subjects[251:]
    
    os.makedirs(os.path.join(data_root,'BraTS2D'), exist_ok = True)
    for dir in ['train','val','test']:
        os.makedirs(os.path.join(data_root,'BraTS2D',dir), exist_ok = True)
        os.makedirs(os.path.join(data_root,'BraTS2D',dir,'images'), exist_ok = True)
        os.makedirs(os.path.join(data_root,'BraTS2D',dir,'masks'), exist_ok = True)
    
    train_data = os.path.join(data_root, "BraTS2D/train/images")
    train_mask = os.path.join(data_root, "BraTS2D/train/masks")
    val_data = os.path.join(data_root, "BraTS2D/val/images")
    val_mask = os.path.join(data_root, "BraTS2D/val/masks")
    test_data =  os.path.join(data_root, "BraTS2D/test/images")
    test_mask =  os.path.join(data_root, "BraTS2D/test/masks")
    train_counts = 0
    val_counts = 0
    test_counts = 0
    # for each subjects in training, add normal slices to training and abnormal slices to validation
    def store_image_from_array(image_folder,mask_folder,subject,image,mask,slice_idx):
        slice_image = Image.fromarray(image, mode='L')
        slice_mask = Image.fromarray(mask, mode='L')
        png_filename = os.path.join(image_folder, '{}_{}{}'.format(subject,slice_idx,".png"))
        mask_filename = os.path.join(mask_folder, '{}_{}{}'.format(subject,slice_idx,".png"))
        slice_image.save(png_filename)
        slice_mask.save(mask_filename)
    def normalize_array(array):
        return array
        if np.ptp(array) == 0:
            return np.zeros_like(array, dtype='uint8')
        else: 
            array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
            return array.astype(np.uint8)
    for subject in train_subjects:
        volume_path = os.path.join(IXI_path, subject)
        volume = nib.load(volume_path).get_fdata()
        volume = crop_3d_volume_to_size(volume, resize_size=volume_size,target_size=volume_size)
        if np.ptp(volume) == 0:
            volume = np.zeros_like(volume, dtype='uint8')
        else:
            volume = ((volume-volume.min())/(volume.max()-volume.min())*255).astype('uint8')
        mask = np.zeros_like(volume,dtype='uint8')*255
        for slice_idx in range(volume.shape[2]):
            train_counts += 1
            slice = normalize_array(volume[:,:,slice_idx])
            slice_mask = mask[:,:,slice_idx].astype('uint8')
            store_image_from_array(train_data,train_mask,subject, slice, slice_mask,slice_idx)
    
    for subject in val_subjects:
        volume_path = os.path.join(BraTS_image_path, subject+"_t1.nii.gz")
        mask_path = os.path.join(BraTS_mask_path, subject+"_seg.nii.gz")
        volume = nib.load(volume_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        volume = crop_3d_volume_to_size(volume, resize_size=volume_size,target_size=volume_size)
        if np.ptp(volume) == 0:
            volume = np.zeros_like(volume, dtype='uint8')
        else:
            volume = ((volume-volume.min())/(volume.max()-volume.min())*255).astype('uint8')
        mask =  crop_3d_volume_to_size(mask, resize_size=volume_size,target_size=volume_size)
        if np.ptp(mask) == 0:
            mask = np.zeros_like(mask, dtype='uint8')
        else:
            mask = ((mask-mask.min())/(mask.max()-mask.min())*255).astype('uint8')
        for slice_idx in range(volume.shape[2]):
            val_counts += 1
            slice = normalize_array(volume[:,:,slice_idx])
            slice_mask = mask[:,:,slice_idx]
            store_image_from_array(val_data,val_mask,subject, slice, slice_mask,slice_idx)
    for subject in test_subjects:
        volume_path = os.path.join(BraTS_image_path,subject+"_t1.nii.gz")
        mask_path = os.path.join(BraTS_mask_path, subject+"_seg.nii.gz")
        volume = nib.load(volume_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        volume = crop_3d_volume_to_size(volume, resize_size=volume_size,target_size=volume_size)
        if np.ptp(volume) == 0:
            volume = np.zeros_like(volume, dtype='uint8')
        else:
            volume = ((volume-volume.min())/(volume.max()-volume.min())*255).astype('uint8')
        mask =  crop_3d_volume_to_size(mask, resize_size=volume_size,target_size=volume_size)
        if np.ptp(mask) == 0:
            mask = np.zeros_like(mask, dtype='uint8')
        else:
            mask = ((mask-mask.min())/(mask.max()-mask.min())*255).astype('uint8')
        for slice_idx in range(volume.shape[2]):
            test_counts += 1
            slice = normalize_array(volume[:,:,slice_idx])
            slice_mask = mask[:,:,slice_idx]
            store_image_from_array(test_data,test_mask,subject, slice, slice_mask,slice_idx)
    print('train_counts:',train_counts)
    print('val_counts:',val_counts)
    print('test_counts:', test_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output directories.")
    parser.add_argument('--data_base_dir', type=str, required=True, help="Path to the data base directory")
    args = parser.parse_args()

    T1_data_path = os.path.join(args.data_base_dir,'T1')
    T2_data_path = os.path.join(args.data_base_dir, 'T2')
    IXIT1_path = os.path.join(args.data_base_dir, 'IXI-T1')
    IXIT2_path = os.path.join(args.data_base_dir, 'IXIT2')
    BraTS_data_path = os.path.join(args.data_base_dir, 'BraTS2021')
    
    T1_temp = os.path.join(T1_data_path, 'temp')
    T2_temp = os.path.join(T2_data_path, 'temp')
    os.makedirs(T1_temp, exist_ok=True)
    os.makedirs(T2_temp, exist_ok=True)

    IXIT1_skull_stripped_path = os.path.join(T1_data_path, 'IXI_skull-stripped')
    IXIT1_reversed_path = os.path.join(T1_data_path, 'IXI_reversed')
    IXIT1_registered_path = os.path.join(T1_data_path, 'IXI_registered')
    T1_base_template =  './../../fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
    #T1_base_template = os.path.join(IXIT1_path,'IXI002-Guys-0828-T1.nii.gz')

    
    IXIT2_skull_stripped_path = os.path.join(T2_data_path, 'IXI_skull-stripped')
    IXIT2_registered_path = os.path.join(T2_data_path, 'IXI_registered')
    T2_base_template = './../../fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
    #T2_base_template = os.path.join(IXIT2_path, 'IXI059-HH-1284-T2.nii.gz')

    # step 1: process BraTS2021
    print('------------------ BraTS2021 folder process -------------------')
    #process_BraTS(BraTS_data_path, T1_data_path,'T1')
    #print('Successfully done process on BraTST1')
    #process_BraTS(BraTS_data_path, T2_data_path,'T2')
    #print('Successfully done process on BraTST2')
    # step 2: perform brain extraction: skull-stripped
    print(('-------------------- brain extraction --------------------'))
    #brain_extraction(IXIT1_path, IXIT1_skull_stripped_path)
    #print('Successfully done brain extraction on IXIT1')
    #brain_extraction(IXIT2_path, IXIT2_skull_stripped_path)
    #print('Successfully done brain extraction on IXIT2')

    # step 3: coregistered them to the same template with 1mm
    print('-------------------- co-registeration --------------------')
    #registered_nii_IXIT1(T1_base_template, IXIT1_skull_stripped_path, IXIT1_registered_path, IXIT1_reversed_path)
    #print('Successfully done registeration on IXIT1')
    #registered_nii_IXIT2(T2_base_template, IXIT2_skull_stripped_path, IXIT2_registered_path)
    #print('Successfully done registeration on IXIT2')
    #registered_nii_BraTS(T1_base_template, T1_data_path,'T1')
    #print('Successfully done registeration on BraTST1')
    #registered_nii_BraTS(T2_base_template, T2_data_path,'T2')
    #print('Successfully done registeration on BraTST2')

    # step 4: create 2D data
    #print('-------------------- load 2D slices --------------------')
    load_slices_for_3D(T1_data_path, BraTS_data_path, (182,218,182))
    print('Successfully load 2D T1 slices')
    load_slices_for_3D(T2_data_path, BraTS_data_path, (182,218,182))
    print('Successfully load 2D T2 slices')
    
