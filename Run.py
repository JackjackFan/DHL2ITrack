import os
from os.path import join, isdir
from tracker import *
import numpy as np
import argparse
import pickle
import math
import warnings
import time

warnings.simplefilter("ignore", UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
# import the_module_that_warns

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)
    RGB_img_list = sorted([seq_path + '/channel/' + p for p in os.listdir(seq_path + '/channel') if os.path.splitext(p)[1] == '.jpg'])
    T_img_list = sorted([seq_path + '/channel2/' + p for p in os.listdir(seq_path + '/channel2') if os.path.splitext(p)[1] == '.jpg'])

    RGB_gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')
    T_gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_MDNet():

    ## option setting
    opts['model_path']= 'models/DHLITrack_v2_rt_50_pr95.pth'###v6 结果的模型
    opts['visualize'] = False
    
    ## for GTOT
    opts['lr_init'] = 0.00035
    opts['lr_update'] = 0.0002
    opts['lr_mult'] = {'fc6':11}
    opts['maxiter_update'] = 10 
    opts['maxiter_init'] = 65 
    opts['trans_f_expand'] = 1.4

    model_name = opts['model_path'].split('/')[-1]

    ## path initialization
    dataset_path = '/media/zmcv/data2/LOLT156/low-light_infrared_dataset_test/'


    seq_home = dataset_path 
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    seq_list.sort()
    fps_record = []
    select_video = ['car5','corridor1','courtyard2','lightup','toycar1','twoman1','truck1','car8']
    for num,seq in enumerate(seq_list):
        save_path = './results/' + 'my_low_light' +  '/' + seq + '.txt'
        save_folder = './results/' + 'my_low_light'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if os.path.exists(save_path):
            continue
        if num<-1:
            continue
        if seq not in select_video:
            continue
        seq_path = seq_home + '/' + seq
        print('——————————Process sequence: '+seq +'——————————————')
        RGB_img_list, T_img_list, RGB_gt, T_gt =genConfig(seq_path,None)
        result, fps = run_mdnet(RGB_img_list, T_img_list, RGB_gt[0], RGB_gt, seq = seq, display=opts['visualize'])
        print ('{} {} , fps:{}'.format(num,seq, fps))
        np.savetxt(save_path,result,fmt = '%d',delimiter = ',')
        fps_record.append(fps)
    if len(fps_record):
        print(sum(fps_record)/len(fps_record))


if __name__ =='__main__':

    run_MDNet()
