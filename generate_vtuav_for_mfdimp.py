import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'modules')
from modules.data_prov import *
from modules.model import *
from modules.pretrain_options import *
from tracker import *
import numpy as np
from modules.fsh_retinex_model import RetinexNet

import argparse
import cv2

def set_optimizer_train(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'], w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer



def train_mdnet():

    ## set image directory
    img_home = '/media/zmcv/VTUAV/train/ST'
    fusion_model = RetinexNet()

    if pretrain_opts['use_gpu']:
        fusion_model = fusion_model.cuda()        
    fusion_model.fusion_init()


    lolt_mfdim_save_path = "/media/zmcv/VTUAV/train/ST"

    with open(os.path.join(img_home,'vtuav_train_st.txt')) as f:
        video_list = f.readlines()
    # video_list = sorted(os.listdir(img_home))
    
    for i,video_name in enumerate(video_list):
        print('{}---now is processing {}'.format(i,video_name.strip()))
        video_name = os.path.join(img_home,video_name.strip())
        lowlight_img_list = sorted(os.listdir(os.path.join(img_home,video_name,'rgb')))
        T_img_list = sorted(os.listdir(os.path.join(img_home,video_name,'ir')))
        seqlen = len(lowlight_img_list)
        for j,(ll_img_file,t_img_file) in enumerate(zip(lowlight_img_list,T_img_list)):
            ll_img = np.array(Image.open(os.path.join(img_home,video_name,'rgb',ll_img_file)).convert("L"))
            t_img = np.array(Image.open(os.path.join(img_home,video_name,'ir',t_img_file)))      

            ll_img = cv2.resize(ll_img,None,fx=0.5,fy=0.5)
            t_img = cv2.resize(t_img,None,fx=0.5,fy=0.5)

            channel_1,channel_2,channel_3= fusion_model.fusion_process(ll_img,t_img[:,:,0])
            if not os.path.exists(os.path.join(lolt_mfdim_save_path,video_name,'enhance')):
                os.mkdir(os.path.join(lolt_mfdim_save_path,video_name,'enhance'))
                os.mkdir(os.path.join(lolt_mfdim_save_path,video_name,'fusion'))
            channel_1_sava_path = os.path.join(lolt_mfdim_save_path,video_name,'enhance',ll_img_file+'_enhance.jpg')
            channel_3_sava_path = os.path.join(lolt_mfdim_save_path,video_name,'fusion',ll_img_file+'_fusion.jpg')

            ll_img = cv2.resize(ll_img,None,fx=2,fy=2)
            t_img = cv2.resize(t_img,None,fx=2,fy=2)            
            cv2.imwrite(channel_1_sava_path,channel_1)
            cv2.imwrite(channel_3_sava_path,channel_3)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'VOT' )
    parser.add_argument("-padding_ratio", default = 5., type =float)
    # parser.add_argument("-model_path", default =".models/rt_mdnet.pth", help = "model path")
    parser.add_argument("-frame_interval", default = 1, type=int, help="frame interval in batch. ex) interval=1 -> [1 2 3 4 5], interval=2 ->[1 3 5]")
    parser.add_argument("-init_model_path", default="/home/cv/data1/4st-RGBT_tracking/RT-MDNet-micro_infrared_base/models/rt-mdnet.pth")
    parser.add_argument("-batch_frames", default = 8, type = int)
    parser.add_argument("-lr", default=0.0001, type = float)
    parser.add_argument("-batch_pos",default = 64, type = int)
    parser.add_argument("-batch_neg", default = 196, type = int)
    parser.add_argument("-n_cycles", default = 200, type = int )
    parser.add_argument("-adaptive_align", default = True, action = 'store_false')
    parser.add_argument("-seqbatch_size", default=50, type=int)

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ##option setting
    pretrain_opts['set_type'] = args.set_type
    pretrain_opts['padding_ratio']=args.padding_ratio
    pretrain_opts['padded_img_size']=pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
    # pretrain_opts['model_path']=args.model_path
    pretrain_opts['frame_interval'] = args.frame_interval
    #pretrain_opts['init_model_path'] = args.init_model_path
    pretrain_opts['batch_frames'] = args.batch_frames
    pretrain_opts['lr'] = args.lr
    pretrain_opts['batch_pos'] = args.batch_pos  # original = 64
    pretrain_opts['batch_neg'] = args.batch_neg  # original = 192
    #pretrain_opts['n_cycles'] = args.n_cycles
    pretrain_opts['adaptive_align']=args.adaptive_align
    pretrain_opts['seqbatch_size'] = args.seqbatch_size
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################

    print(pretrain_opts)
    train_mdnet()

