import argparse
import ast
import torch

from dataloaders import dataloader_BraTS_2D_setup, dataloader_MVTec_setup
from utils import  load_backbone
from models import MRI_SimpleNet_2D, MRI_CNF_SimpleNet_2D

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
    
    print('args: ',args)
    if args.edc:
        backbone = load_backbone(args.backbone, args.edc)
    else:
        backbone = load_backbone(args.backbone)
    
    torch.cuda.empty_cache()
    if args.score_model == 'disc':
        model = MRI_SimpleNet_2D(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    elif args.score_model == 'flow':
        model = MRI_CNF_SimpleNet_2D(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    else:
        print('wrong score model type')
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
    parser.add_argument("--sub_dir", type=str, default="BraTS")
    parser.add_argument("--run_name", type=str, default='final')
    parser.add_argument("--run_info", type=str, default="final testing")
    # data setup
    parser.add_argument("--volume_size", type=parse_tuple, default="(256,256)")
    parser.add_argument("--resize_size", type=parse_tuple, default="(256,256)")
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--dataset", type=str, default='BraTS')
    parser.add_argument("--class_name", type=str, default="screw")
    # feature extraction setup
    parser.add_argument("--backbone", type=str,default="wide_resnet50_2")
    parser.add_argument("--patch_size",type=int, default=3)
    parser.add_argument("--edc", action="store_true")

    # noise settup
    parser.add_argument("--noise_type", type=str, default="simple")
    parser.add_argument("--noise_std", type=float, default=0.06)
    parser.add_argument("--noise_res", type=int, default=4)
    parser.add_argument("--mix_noise", type=int, default=1)
    
    # projection setup
    parser.add_argument("--proj_layers", type=int, default=0)
    parser.add_argument("--preprocessing_dimension", type=int, default=512)
    # discriminator setup
    parser.add_argument("--score_model", type=str, default='flow')
    parser.add_argument("--target_embed_dimension", type=int, default=512)
    parser.add_argument("--dsc_margin", type=float, default=0.5)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)

    # conditional normalizing flow setup
    parser.add_argument("--flow_lr", type=float, default=1e-3)
    parser.add_argument("--flow_lr_step",type=int, default=10)
    parser.add_argument("--coupling_layers", type=int, default=2)
    parser.add_argument("--flow_hidden", type=int, default=1024)
    parser.add_argument("--pos_embed_dim", type=int, default='128')
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_beta",type=float,default=0.05)
    parser.add_argument("--margin_tau",type=float,default=0.1)
    parser.add_argument("--normalizer",type=int, default=10)

    # training setup
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_per_epochs",type=int,default=10)
    parser.add_argument("--stop_gradient",action="store_true",help="perform stop-gradient operation")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--use_attention", action="store_true")

    # learning rate
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dsc_lr", type=float, default=2*1e-4)
    args = parser.parse_args(["--use_gpu"])
    main(args)