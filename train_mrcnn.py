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
    img_home = '/media/zmcv/data2/LOLT156/low-light_infrared_dataset_test/'
    data_path = 'DATA/DHLItrack_v2_50.pkl'

    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)


    K = len(data)


    print(K)
    fusion_model = RetinexNet()
    model = MDNet(pretrain_opts['init_model_path'], K,train= True)
    if pretrain_opts['adaptive_align']:
        align_h = model.roi_align_model.pooled_height
        align_w = model.roi_align_model.pooled_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = PrRoIPool2D(align_h, align_w, spatial_s)

    if pretrain_opts['use_gpu']:
        model = model.cuda()
        fusion_model = fusion_model.cuda()        
    model.set_learnable_params(pretrain_opts['ft_layers'])
    model.train()
    fusion_model.fusion_init()



    dataset = [None] * K
    for k, (seqname, seq) in enumerate(data.items()):
        RGB_img_list = seq['RGB_image']
        T_img_list = seq['T_image']
        RGB_gt = seq['RGB_gt']
        T_gt = seq['T_gt']

        img_dir = img_home + seqname
        
        dataset[k]=RegionDataset(img_dir,RGB_img_list,T_img_list,RGB_gt,T_gt,model.receptive_field,pretrain_opts)



    ## Init criterion and optimizer ##
    binaryCriterion = BinaryLoss()
    interDomainCriterion = nn.CrossEntropyLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, pretrain_opts['lr'])

    best_score = 0.
    batch_cur_idx = 0
    for i in range(pretrain_opts['n_cycles']):
        print ("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        totalTripleLoss = np.zeros(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            #RGB_cropped_scenes,T_cropped_scenes, pos_rois, neg_rois, init_RGB_targets, init_T_targets= dataset[k].next()

            try:
                RGB_image_list,T_image_list,init_RGB_image,init_T_image,sample_idx = dataset[k].next()
            except:
                continue
            input_img = []
            channel_1,channel_2,channel_3= fusion_model.fusion_process(init_RGB_image,init_T_image)
            init_img = np.concatenate((channel_1,channel_2,channel_3),2)
            for i in range(0,len(RGB_image_list)):
                channel_1,channel_2,channel_3= fusion_model.fusion_process(RGB_image_list[i],T_image_list[i])
                input_img.append(np.concatenate((channel_1,channel_2,channel_3),2))
            # RGB_cropped_scenes,T_cropped_scenes, pos_rois, neg_rois, init_RGB_targets, init_T_targets
            RGB_cropped_scenes,pos_rois, neg_rois, init_RGB_targets = dataset[k].tracking_process(input_img,init_img,sample_idx)
            for sidx in range(0, len(RGB_cropped_scenes)):
                RGB_cur_scene = RGB_cropped_scenes[sidx]
                init_RGB_target = init_RGB_targets[sidx]
                cur_pos_rois = pos_rois[sidx]
                cur_neg_rois = neg_rois[sidx]

                if cur_pos_rois.data.shape == torch.Size([0]):
                    continue
                RGB_cur_scene = Variable(RGB_cur_scene)
                cur_pos_rois = Variable(cur_pos_rois)
                cur_neg_rois = Variable(cur_neg_rois)
                if pretrain_opts['use_gpu']:
                    RGB_cur_scene = RGB_cur_scene.cuda()
                    cur_pos_rois = cur_pos_rois.cuda()
                    cur_neg_rois = cur_neg_rois.cuda()
                    init_RGB_target = init_RGB_target.cuda()

                cur_feat_map = model(RGB_cur_scene, k, out_layer='conv3')

                cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                if sidx == 0:
                    pos_feats = []
                    neg_feats = []
                    if cur_pos_feats.shape[0] < 64:
                        diff = 64-cur_pos_feats.shape[0]
                        times = diff//cur_pos_feats.shape[0]
                        for ii in range(0,times+1):
                            cur_pos_feats=torch.cat((cur_pos_feats,cur_pos_feats[-diff:]))
                            diff = 64-cur_pos_feats.shape[0]
                    if cur_pos_feats.shape[0] > 64:
                        cur_pos_feats = cur_pos_feats[0:64]
                    if cur_pos_feats.shape[0] != 64:
                        print(1)
                    pos_feats.append(cur_pos_feats)
                    neg_feats.append(cur_neg_feats)
                else:
                    if cur_pos_feats.shape[0]<64:
                        diff = 64-cur_pos_feats.shape[0]
                        times = diff//cur_pos_feats.shape[0]
                        for ii in range(0,times+1):
                            cur_pos_feats=torch.cat((cur_pos_feats,cur_pos_feats[-diff:]))
                            diff = 64-cur_pos_feats.shape[0]
                    if cur_pos_feats.shape[0] > 64:
                        cur_pos_feats = cur_pos_feats[0:64]

                    if cur_pos_feats.shape[0] != 64:
                        print(1)
                    pos_feats.append(cur_pos_feats)
                    neg_feats.append(cur_neg_feats)
            feat_dim = cur_neg_feats.size(1)
            pos_feats1 = torch.stack(pos_feats,dim=0).view(-1,feat_dim)
            neg_feats1 = torch.stack(neg_feats,dim=0).view(-1,feat_dim)

            pos_score = model(pos_feats1, k, in_layer='fc4')
            neg_score = model(neg_feats1, k, in_layer='fc4')

            cls_loss = binaryCriterion(pos_score, neg_score)

            ## inter frame classification

            interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
            if opts['use_gpu']:
                interclass_label = interclass_label.cuda()
            total_interclass_score = pos_score[:,1].contiguous()
            total_interclass_score = total_interclass_score.view((pos_score.size(0),1))

            K_perm = np.random.permutation(K)
            K_perm = K_perm[0:100]
            for cidx in K_perm:
                if k == cidx:
                    continue
                else:
                    interclass_score = model(pos_feats1, cidx, in_layer='fc4')
                    total_interclass_score = torch.cat((total_interclass_score,interclass_score[:,1].contiguous().view((interclass_score.size(0),1))),dim=1)

            interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
            totalInterClassLoss[k] = interclass_loss.item()

            (cls_loss+0.1*interclass_loss).backward()

            batch_cur_idx+=1
            if (batch_cur_idx%pretrain_opts['seqbatch_size'])==0:
                torch.nn.utils.clip_grad_norm(model.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
                batch_cur_idx = 0
            ## evaulator
            prec[k] = evaluator(pos_score, neg_score)
            ## computation latency
            toc = time.time() - tic
      

        cur_score = prec.mean()
        try:
            total_miou = sum(total_iou)/len(total_iou)
        except:
            total_miou = 0.
        print ("Mean Precision: %.3f Inter Loss: %.3f IoU: %.3f" % (prec.mean(), totalInterClassLoss.mean(),total_miou))
        if cur_score > best_score:
            best_score = cur_score
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print("Save model to %s" % pretrain_opts['model_path'])
            torch.save(states, pretrain_opts['model_path'])
            if pretrain_opts['use_gpu']:
                model = model.cuda()


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

