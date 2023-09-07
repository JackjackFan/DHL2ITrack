import os
import numpy as np
import pickle
from collections import OrderedDict



# seq_home = '../dataset/'
seqlist_path = 'lolt156.txt'
output_path_low_light = 'DATA/DHLItrack_v2_50.pkl'
output_path_infrared = 'DATA/DHLItrack_v2_train_infrared_list.pkl'
seq_home = '/media/zmcv/data2/LOLT156/low-light_infrared_dataset_test/'


# with open(output_path_low_light, 'rb') as fp1:
#     data1 = pickle.load(fp1)

# with open(output_path_infrared, 'rb') as fp1:
#     data2 = pickle.load(fp1)    



with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

low_data = {}
infra_data = {}
for i,seqname in enumerate(seq_list):
    if '.' in seqname:
        continue
    print(seqname)
    seq_path = seq_home+seqname
    low_light_img_list = sorted([p for p in os.listdir(seq_path+'/channel') if os.path.splitext(p)[1] == '.jpg'])
    low_infra_img_list = sorted([p for p in os.listdir(seq_path+'/channel2') if os.path.splitext(p)[1] == '.jpg'])    

    # if (seqname == 'Jogging') or (seqname == 'Skating2'):
    #     gt = np.loadtxt(seq_path + '/groundtruth_rect.1.txt')
    # elif seqname == 'Human4' :
    #     gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt', delimiter=',')
    # elif (seqname == 'BlurBody') or (seqname == 'BlurCar1') or (seqname == 'BlurCar2') or (seqname == 'BlurCar3') \
    #         or (seqname == 'BlurCar4') or (seqname == 'BlurFace') or (seqname == 'BlurOwl') or (seqname == 'Board') \
    #         or (seqname == 'Box') or (seqname == 'Car4') or (seqname == 'CarScale') or (seqname == 'ClifBar') \
    #         or (seqname == 'Couple') or (seqname == 'Crossing') or (seqname == 'Dog') or (seqname == 'FaceOcc1') \
    #         or (seqname == 'Girl') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
    #         or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
    #         or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman') :
    #     gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
    # elif (seqname == 'Diving'):
    #     gt = np.loadtxt(seq_path + '/groundtruth_rect_ilchae.txt', delimiter=',')
    # else:
    gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

    # if (seqname == 'David') or (seqname == 'Football1') or (seqname == 'Freeman3') or (seqname == 'Freeman4'):
    #     continue

    # if set_type =='VOT':
    #     img_list = sorted([p for p in os.listdir(seq_home + seqname) if os.path.splitext(p)[1] == '.jpg'])
    #     gt = np.loadtxt(seq_home + seqname + '/groundtruth.txt', delimiter=',')

    # if set_type == 'IMAGENET':
    #     img_list = []
    #     gt = []
    
    assert len(low_light_img_list) == len(gt), "Lengths do not match!!"

    if gt.shape[1]==8:
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    low_data[seqname] = {'RGB_image':low_light_img_list, 'T_image':low_infra_img_list,'RGB_gt':gt,'T_gt':gt}
    infra_data[seqname] = {'images':low_infra_img_list, 'gt':gt}    

with open(output_path_low_light, 'wb') as fp:
    pickle.dump(low_data, fp, -1)
# with open(output_path_infrared, 'wb') as fp:
#     pickle.dump(infra_data, fp, -1)    
