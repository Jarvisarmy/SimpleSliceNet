from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os

from datasets import MRIDataset_2D, MRIDataset_3D,Br35H, MVTecDataset
from datasets.mvtec import DatasetSplit

def dataloader_MVTec_setup(data_root,
                           class_name,
                           resize_size = (256,256),
                           target_size = (224,224),
                           batch_size=16):
    data_path = os.path.join(data_root, 'mvtec')
    train_dataset = MVTecDataset(data_path, class_name, resize=resize_size, imagesize=target_size,split=DatasetSplit.TRAIN,
        train_val_split=1.0)
    test_dataset = MVTecDataset(data_path, class_name, resize=resize_size, imagesize=target_size,split=DatasetSplit.TEST)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': None,
        'test_loader': test_loader
    }

def dataloader_BraTS_2D_setup(data_root,
                              resize_size=(256,256),
                              target_size=(224,224),
                              batch_size=16):
    train_data = []
    val_data = []
    test_data = []
    train_mask = []
    val_mask = []
    test_mask = []
    data_path = os.path.join(data_root, 'BraTS2D')
    for file in os.listdir(os.path.join(data_path,'train','images')):
        train_data.append(os.path.join(data_path, 'train','images',file))
        train_mask.append(os.path.join(data_path, 'train','masks',file))
    for file in os.listdir(os.path.join(data_path,'val','images')):
        val_data.append(os.path.join(data_path, 'val','images',file))
        val_mask.append(os.path.join(data_path, 'val','masks',file))
    for file in os.listdir(os.path.join(data_path,'test','images')):
        test_data.append(os.path.join(data_path, 'test','images',file))
        test_mask.append(os.path.join(data_path, 'test','masks',file))
    #train_data, train_mask = train_data[:5000], train_mask[:5000]
    #test_data, test_mask = test_data[:5000], test_mask[:5000]
    print('training: {}, validation: {}, testing: {}'.format(len(train_data),
                                                             len(val_data),
                                                             len(test_data)))
    train_dataset = MRIDataset_2D(train_data,train_mask, resize_size=resize_size,target_size=target_size)
    val_dataset = MRIDataset_2D(val_data, val_mask, resize_size=resize_size,target_size=target_size)
    test_dataset = MRIDataset_2D(test_data, test_mask, resize_size=resize_size,target_size=target_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }
def dataloader_IXI_BraTS_2D_setup(data_root,
                                  resize_size=(224,224), 
                                  target_size=(224,224),
                                  test_size=1,
                                  batch_size=16,
                                  IXI_path = 'IXI_2D',
                                  BraTS_path = 'BraTS_2D',
                                  BraTS_mask_path = 'BraTS_mask_2D'):
    train_data = []
    for file in os.listdir(os.path.join(data_root,IXI_path)):
        if file.startswith('IXI'):
            train_data.append(os.path.join(data_root,IXI_path, file))

    test_data = []
    test_mask = []
    for file in os.listdir(os.path.join(data_root,BraTS_path)):
        if file.startswith('BraTS2021'):
            test_data.append(os.path.join(data_root,BraTS_path,file))
            test_mask.append(os.path.join(data_root,BraTS_mask_path,file.replace('t2','seg')))
    combined_files = list(zip(test_data,test_mask))
    if test_size < 1:
        val_files, test_files = train_test_split(combined_files,test_size=test_size,random_state=1)
    val_data, val_mask = zip(*val_files)
    test_data, test_mask = zip(*test_files)

    train_dataset = MRIDataset_2D(train_data,
                                  resize_size=resize_size,
                                  target_size=target_size,
                                  )
    test_dataset = MRIDataset_2D(test_data,
                                 test_mask,
                                  resize_size=resize_size,
                                  target_size=target_size,
                                  )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if test_size < 1:
        val_dataset = MRIDataset_2D(val_data,
                                val_mask,
                                  resize_size=resize_size,
                                  target_size=target_size,
                                  )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }

def dataloader_Br35H_setup(data_root,batch_size):
    data_path = os.path.join(data_root,'Br35H')
    train_dset = Br35H(name='br35h_train', train=True, data_dir=data_path)
    train_dset = train_dset.get_dset()
    print('TrainSet Image Number:', len(train_dset))
    eval_dset = Br35H(name='br35h_test', train=False, data_dir=data_path)
    eval_dset = eval_dset.get_dset()
    print('EvalSet Image Number:', len(eval_dset))
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(eval_dset, batch_size=batch_size, shuffle=False)
    return {
        'train_loader': train_loader,
        'val_loader': None,
        'test_loader': test_loader
    }
    
def dataloader_IXI_BraTS_setup(data_root, resize_size,volume_size,batch_size):
    train_data = []
    for file in os.listdir(os.path.join(data_root,'IXI_registered')):
        if file.startswith('IXI'):
            train_data.append(os.path.join(data_root,'IXI_registered', file))

    test_data = []
    test_mask = []
    for file in os.listdir(os.path.join(data_root,'BraTS_registered')):
        if file.startswith('BraTS2021'):
            test_data.append(os.path.join(data_root,'BraTS_registered',file))
            test_mask.append(os.path.join(data_root,'BraTS_mask_registered',file.replace('t2.nii.gz','seg.nii.gz')))
    
    #val_data = train_data[500:] + test_data[:50]
    #val_mask = [None]*(len(train_data)-500) + test_mask[:50]

    #train_data = train_data[:100]

    test_data = test_data[:150]
    test_mask = test_mask[:150]

    train_dataset = MRIDataset_3D(train_data, resize_size=resize_size,volume_size=volume_size)
    test_dataset = MRIDataset_3D(test_data, test_mask,resize_size=resize_size, volume_size = volume_size)

    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }

    