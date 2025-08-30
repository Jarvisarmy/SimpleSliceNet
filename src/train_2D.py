import argparse
import ast
import torch

from dataloaders import dataloader_BraTS_2D_setup, dataloader_MVTec_setup
from utils import  load_backbone
from models import MRI_SimpleNet_2D, MRI_CNF_SimpleNet_2D_triplet
from runners.Br35H_studies import study_architecture

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print('device: {}'.format(device))
    if args.dataset == 'BraTS':
        dataloaders = dataloader_BraTS_2D_setup(args.data_root,
                                            resize_size = args.resize_size,
                                            target_size = args.volume_size,
                                            batch_size = args.batch_size)
    elif args.dataset == 'MVTec':
        dataloaders = dataloader_MVTec_setup(args.data_root,
                                             args.class_name,
                                             resize_size = args.resize_size,
                                            target_size = args.volume_size,
                                            batch_size = args.batch_size)
    print('train_size: {}, val_size: {}, test_size: {}'.format(
        len(dataloaders['train_loader'].dataset),
        len(dataloaders['val_loader'].dataset) if dataloaders['val_loader'] else 0,
        len(dataloaders['test_loader'].dataset)))
    
    
    # train with discriminator only
    if args.score_model=='flow':
        args.pos_embed_dim = 128
        args.clamp_alpha = 1.9
        args.coupling_layers = 8

        args.pos_beta = 0.05
        args.margin_tau = 0.1
        args.normalizer = 10
    print('args: ',args)
    if args.edc:
        backbone = load_backbone(args.backbone, args.edc)
    else:
        backbone = load_backbone(args.backbone)
    
    torch.cuda.empty_cache()
    
    model = MRI_SimpleNet_2D(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    model.train_model(dataloaders['train_loader'], dataloaders['test_loader'],args.epochs)


if __name__ == '__main__':
    def parse_tuple(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError(f"Invalid tuple: {value}")

    parser = argparse.ArgumentParser()
    # environment setup
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    # path setup
    parser.add_argument("--data_root", type=str, default= "./../data")
    parser.add_argument("--result_path", type=str, default="./../results")
    parser.add_argument("--model_dir", type=str, default="MRI_SimpleNet")
    parser.add_argument("--sub_dir", type=str, default="MVTec")
    parser.add_argument("--run_name", type=str, default='original')
    parser.add_argument("--run_info", type=str, default="trianing")
    # data setup
    parser.add_argument("--volume_size", type=parse_tuple, default="(256,256)")
    parser.add_argument("--resize_size", type=parse_tuple, default="(256,256)")
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--dataset", type=str, default='MVTec')
    parser.add_argument("--class_name", type=str, default="screw")
    # feature extraction setup
    parser.add_argument("--backbone", type=str,default="wide_resnet50_2")
    parser.add_argument("--patch_size",type=int, default=3)
    parser.add_argument("--edc", action="store_true")

    # noise settup
    parser.add_argument("--noise_type", type=str, default="simple")
    parser.add_argument("--noise_std", type=float, default=0.015)
    parser.add_argument("--noise_res", type=int, default=4)
    parser.add_argument("--mix_noise", type=int, default=1)
    
    # projection setup
    parser.add_argument("--proj_layers", type=int, default=1)
    parser.add_argument("--preprocessing_dimension", type=int, default=1536)
    # discriminator setup
    parser.add_argument("--score_model", type=str, default='flow')
    parser.add_argument("--target_embed_dimension", type=int, default=1536)
    parser.add_argument("--dsc_margin", type=float, default=0.5)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)
    # training setup
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_per_epochs",type=int,default=10)
    parser.add_argument("--stop_gradient",action="store_true",help="perform stop-gradient operation")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--use_attention", action="store_true")

    # learning rate
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dsc_lr", type=float, default=2*1e-4)
    args = parser.parse_args()
    main(args)
    # run1: simple noise with edc
    # I_auroc: 0.7494501841243862, 
    # I_auprc: 0.8523747133269952, 
    # p_auroc: 0.8699525522853864, 
    # p_auprc: 0.40928890355510006, 
    # thres: 0.042591825127601624, 
    # acc: 0.696, 
    # f1: 0.776798825256975, 
    # recall: 0.8477564102564102, 
    # specificity: 0.4441489361702128,

    # run2: simple noise with edc and tanh
    # I_auroc: 0.7258762956901255, 
    # I_auprc: 0.8329679659898102, 
    # p_auroc: 0.8809686275599974, 
    # p_auprc: 0.40257346294642504, 
    # thres: 0.10684715211391449, 
    # acc: 0.653, 
    # f1: 0.777421423989737, 
    # recall: 0.9711538461538461, 
    # specificity: 0.125,
    
    # run3, coarse noise, res=4, edc
    # I_auroc: 0.7402780619203492, 
    # I_auprc: 0.8357200476366256, 
    # p_auroc: 0.8539331770377689, 
    # p_auprc: 0.3686434735747797, 
    # thres: 0.03471331298351288, 
    # acc: 0.69, 
    # f1: 0.786206896551724, 
    # recall: 0.9134615384615384, 
    # specificity: 0.3191489361702128,

    # run4: simple 
    # I_auroc: 0.7119050054555374, I_auprc: 0.8021443979197606, p_auroc: 0.8885748859054812, p_auprc: 0.3717277087156561, thres: 0.14360982179641724, acc: 0.659, 
    # f1: 0.7837666455294864, recall: 0.9903846153846154, specificity: 0.10904255319148937,

    # with positional encoding, can improve the model performance a little bit
