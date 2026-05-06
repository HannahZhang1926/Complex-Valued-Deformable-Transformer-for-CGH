import os
import torch
torch.backends.cudnn.enabled = False
from torch.utils.data import Dataset, DataLoader
from utils import MyDataset, propagation, PhysicTransformer,ITV2d
import argparse
from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2
from model import *

parser = argparse.ArgumentParser(description='physic-transformer')

# parallel
parser.add_argument('--gpu_list', default='0')
parser.add_argument('--num_workers', default=0)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Dir
parser.add_argument('--img_dir', type=str, default='./DIV2K_valid_HR')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--log_dir', type=str, default='./log')

# Optimization Setting
parser.add_argument('--batch_size', default=2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--warmup_lr',default=5e-7)
parser.add_argument('--min_lr',default=5e-6)

parser.add_argument('--warmup_epochs',default=5)
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=40, help='epoch number of end training')
parser.add_argument('--seed', type=int, default=1111, help='random seed')

# Network Setting
parser.add_argument('--embed_dim', type=int, default=96, help='Embedding dimension')
parser.add_argument('--num_heads', type=int, default=8, help='number of heads for the transformer network')
parser.add_argument('--attn_dropout', type=float, default=0.0, help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.5, help='hidden layer dropout')
parser.add_argument('--trans_layer_num', type=int, default=2, help='transformer encoder layer num')  # Nx
parser.add_argument('--unfolding_stage_num', type=int, default=1, help='number of unfolding stages')
parser.add_argument('--img_res', default=(256, 256), help='resolution of input image')

# Optical Setting
parser.add_argument('--plane_distance', default=5*1e-2, help='distance from SLM plane to target plane')
parser.add_argument('--wavelength', default=520*1e-9)
parser.add_argument('--slm_size', default=(8*1e-6, 8*1e-6), help='SLM pitch')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list

experiment_name = "20240906_complexattn_softmax_1024_1024_batch1_t"
model_dir = "%s/%s" % (args.model_dir, experiment_name)
log_file_name = "%s/%s.txt" % (args.log_dir, experiment_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if __name__ == '__main__':

    train_img = MyDataset(args.img_dir, args.img_res)
    n_gpu = len(args.gpu_list.split(','))
    train_loader = DataLoader(train_img, batch_size=args.batch_size*n_gpu, shuffle=True, num_workers=args.num_workers)
    
    # device = torch.device("cuda:3,4" if torch.cuda.is_available() else "cpu")
    model = PhysicTransformer(embed_dim=args.embed_dim,
                              num_heads=args.num_heads,
                              attn_dropout=args.attn_dropout,
                              relu_dropout=args.relu_dropout,
                              res_dropout=args.res_dropout,
                              out_dropout=args.out_dropout,
                              layers=args.trans_layer_num,
                              slm_size=args.slm_size,
                              wavelength=args.wavelength,
                              plane_distance=args.plane_distance,
                              stage_num=args.unfolding_stage_num
                              )
    # model = model.to(device)
    model = nn.DataParallel(model).cuda()
    
    if args.start_epoch != 0:
        model.load_state_dict(torch.load('./model/{}/net_params_{}.pkl'.format(experiment_name, args.start_epoch)))
        print('Loading model parameters from ', 'net_params_{}.pkl'.format(args.start_epoch))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_f = torch.nn.MSELoss() #试一下改loss

    for epoch_i in range(args.start_epoch + 1, args.end_epoch + 1):
        num_iter = len(train_loader)
        pbar = tqdm(range(num_iter))

        losses = []
        for iter, batch in enumerate(train_loader):
            # batch = batch.to(device)
            batch = batch.cuda()
            image = batch.clone()
            image = image.permute([0, 3, 1, 2])

            x_init = propagation(image, args.slm_size, args.wavelength, -args.plane_distance)
            x_output = model(x_init, image)
            
            loss_F = loss_f(x_output, image)
            loss_all = loss_F

            optimizer.zero_grad()
            loss_all.requires_grad_(True)
            loss_all.backward()
            optimizer.step()

            losses.append(loss_all.item())
            pbar.set_description("Loss: %.8f" % (loss_all.item()))
            pbar.update()
        pbar.close()

        output_data = "Epoch: %d, Avg Loss: %.8f\n" % (epoch_i, sum(losses)/len(losses))
        output_file = open(log_file_name, 'a')
        output_file.write(output_data)
        output_file.close()

        if epoch_i % 1 == 0:
            print("Save Model to: %s/net_params_%d.pkl" % (model_dir, epoch_i))
            torch.save(model.state_dict(), "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
