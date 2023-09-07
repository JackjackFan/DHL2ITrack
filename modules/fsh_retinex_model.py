import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(2, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        
        #fusion
        self.low_inf_fusion =  nn.Sequential(nn.Conv2d(2, channel//2, kernel_size,
                                        padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel//2, channel//4, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel//4, 1, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())#         
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 2, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:1, :, :])
        L        = torch.sigmoid(outs[:, 1:2, :, :])
        return R, L

class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(2, channel, kernel_size,
                                      padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet  = DecomNet()
        self.RelightNet= RelightNet()

    def forward(self, input_low,input_fusion, input_infrared):
        # Forward DecompNet
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        input_fusion = Variable(torch.FloatTensor(torch.from_numpy(input_fusion))).cuda()
        input_infrared= Variable(torch.FloatTensor(torch.from_numpy(input_infrared))).cuda()        
        R_low, I_low   = self.DecomNet(input_low)
        R_fusion, I_fusion   = self.DecomNet(input_fusion)

        R_fusion_low_infra = self.DecomNet.low_inf_fusion(torch.cat((R_low,input_infrared),dim=1))
        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_fusion_low_infra)

        # Compute losses
        self.recon_loss_low  = F.l1_loss(R_low * I_low,  input_low)
        self.recon_loss_high = F.l1_loss(R_fusion * I_fusion, input_fusion)
        self.recon_loss_mutal_low  = F.l1_loss(R_fusion_low_infra * I_fusion, input_low)
        self.recon_loss_mutal_high = F.l1_loss(R_fusion * I_low, input_fusion)
        self.equal_R_loss = F.l1_loss(R_fusion_low_infra,  R_fusion.detach())
        self.relight_loss = F.l1_loss(R_fusion_low_infra * I_delta, input_fusion)

        self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        self.Ismooth_loss_high  = self.smooth(I_fusion, R_fusion)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_fusion_low_infra)

        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
        self.loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta
        self.infrared_save = input_infrared.detach().cpu()
        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low.detach().cpu()
        self.output_R_fusion_low_infrared   = R_fusion_low_infra.detach().cpu()
        self.output_I_delta = I_delta.detach().cpu()
        self.output_S       = R_fusion_low_infra.detach().cpu() * I_delta.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        # input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        # input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input    = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image= np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                result_5 = self.output_R_fusion_low_infrared
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                result_5 = np.squeeze(result_5)
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_5,result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png'.format((train_phase, idx + 1, epoch_num)))
                      
            im.save(filepath[:-4] + '.jpg')


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir):
        load_dir   = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts)>0:
                load_ckpt  = load_ckpts[-1]
                global_step= int(load_ckpt[:-4])
                ckpt_dict  = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self,
              train_low_data_names,
              train_fusion_data_names,
              train_infrared_data_names,              
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_fusion_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase= train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 1, patch_size, patch_size,), dtype="float32")
                batch_input_fusion = np.zeros((batch_size, 1, patch_size, patch_size,), dtype="float32")
                batch_input_infrared= np.zeros((batch_size, 1, patch_size, patch_size,), dtype="float32")                
                for patch_id in range(batch_size):
                    # Load images
                    try:
                        channel_index = random.choices([0,1,2],[1,1,1])[0]
                        train_low_img = Image.open(train_low_data_names[image_id])
                        train_low_img = np.array(train_low_img, dtype='float32')/255.0

                        train_fusion_img = Image.open(train_fusion_data_names[image_id])
                        train_fusion_img = np.array(train_fusion_img, dtype='float32')/255.0

                        train_infrared_img= Image.open(train_infrared_data_names[image_id])
                        train_infrared_img= np.array(train_infrared_img, dtype='float32')/255.0   
                    except:
                        print(train_low_data_names[image_id])   
                        print(train_fusion_data_names[image_id])  
                        print(train_infrared_data_names[image_id])  

                    train_low_img = np.expand_dims(train_low_img,axis = 2)
                    train_fusion_img = np.expand_dims(train_fusion_img[:,:,channel_index],axis = 2)
                    train_infrared_img = np.expand_dims(train_infrared_img,axis = 2)                    
                    # Take random crops
                    h, w, _        = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_fusion_img = train_fusion_img[x: x + patch_size, y: y + patch_size, :]
                    train_infrared_img= train_infrared_img[x: x + patch_size, y: y + patch_size, :]


                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_fusion_img = np.flipud(train_fusion_img)
                        train_infrared_img= np.flipud(train_infrared_img)                        
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_fusion_img = np.fliplr(train_fusion_img)
                        train_infrared_img= np.fliplr(train_infrared_img)                        
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_fusion_img = np.rot90(train_fusion_img, rot_type)
                        train_infrared_img= np.rot90(train_infrared_img, rot_type)                        
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_fusion_img = np.transpose(train_fusion_img, (2, 0, 1))
                    train_infrared_img= np.transpose(train_infrared_img, (2, 0, 1))                    
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_fusion[patch_id, :, :, :] = train_fusion_img
                    batch_input_infrared[patch_id, :, :, :]= train_infrared_img                    
                    self.input_low = batch_input_low
                    self.input_fusion = batch_input_fusion
                    self.input_infrared= batch_input_infrared                    

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names,train_fusion_data_names,train_infrared_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names ,train_fusion_data_names , train_infrared_data_names= zip(*tmp)


                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,self.input_fusion,  self.input_infrared)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                # self.evaluate(epoch + 1, eval_low_data_names,
                #                 vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)

    def singleScaleRetinex(self,img,sigma):
        a = np.log10(img)
        b = np.log10(cv2.GaussianBlur(img,(0,0),sigma))
        return a-b

    def fusion_init(self,
                ckpt_dir = './ckpts/'):
        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
             print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False

    def fusion_process(self,
                input_low_test,input_infrared_test):
        w,h = input_low_test.shape
        test_low_img   = np.expand_dims(np.array(input_low_test, dtype="float32")/255.0,axis = 2)
        test_low_img   = np.transpose(test_low_img, (2, 0, 1))
        input_low_test = np.expand_dims(test_low_img, axis=0)        

        test_infrared_img   = np.expand_dims(np.array(input_infrared_test, dtype="float32")/255.0,axis = 2)
        test_infrared_img   =np.transpose(test_infrared_img, (2, 0, 1))
        input_infrared_test = np.expand_dims(test_infrared_img, axis=0)         
        self.forward(input_low_test,input_low_test,input_infrared_test)

        self.output_R_low = np.squeeze(self.output_R_low,0)
        self.output_R_fusion_low_infrared = np.squeeze(self.output_R_fusion_low_infrared,0)
        # input_infrared_test = np.squeeze(input_infrared_test,0)
        self.output_S = np.squeeze(self.output_S,0)
        cat_image= np.concatenate([self.output_R_low,self.output_R_fusion_low_infrared ,self.output_S], axis=2)
           
        cat_image = np.transpose(cat_image, (1, 2, 0))
        # print(cat_image.shape)
        tmp1 = np.clip(cat_image * 255.0, 0, 255.0)
        tmp2 = tmp1.astype('uint8')
        tmp2 = np.squeeze(tmp2)     

        return np.expand_dims(tmp2[:,:h],2),np.expand_dims(tmp2[:,h:2*h],2),np.expand_dims(tmp2[:,2*h:3*h],2)

    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        # Load the network with a pre-trained checkpoint
        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception
        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
             print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = True
        
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_name  = test_low_data_names[idx]       
            print('Processing ', test_img_name)
            test_low_img   = Image.open(os.path.join('./data/test/low',test_img_name,'low','000001lll.jpg'))
            # enahnce_tmp = self.singleScaleRetinex(np.minimum(np.maximum(test_low_img,1),255),10)
            # enahnce_tmp[:,:] = (enahnce_tmp[:,:]-np.min(enahnce_tmp[:,:]))/(np.max(enahnce_tmp[:,:])-np.min(enahnce_tmp[:,:]))*255
            # enahnce_tmp = np.array(enahnce_tmp, dtype="uint8")
            test_low_img   = np.expand_dims(np.array(test_low_img, dtype="float32")/255.0,axis = 2)
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            test_infrared_img   = Image.open(os.path.join('./data/test/low',test_img_name,'infrared','000001t.jpg'))
            test_infrared_img   = np.expand_dims(np.array(test_infrared_img, dtype="float32")/255.0,axis = 2)
            test_infrared_img   =np.transpose(test_infrared_img, (2, 0, 1))
            input_infrared_test = np.expand_dims(test_infrared_img, axis=0)            

            self.forward(input_low_test,input_low_test,input_infrared_test)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            result_5 = self.output_R_fusion_low_infrared
            result_6 = self.infrared_save
            input = np.expand_dims(np.squeeze(input_low_test),axis=0)
            result_1 = np.squeeze(result_1,0)
            result_2 = np.squeeze(result_2,0)
            result_3 = np.squeeze(result_3,0)
            result_4 = np.squeeze(result_4,0)
            result_5 = np.squeeze(result_5,0)            
            result_5 = np.squeeze(result_6,0)               
            if save_R_L:
                cat_image= np.concatenate([input, result_1,result_5, result_2, result_3,result_4], axis=2)
            else:
                cat_image= np.concatenate([result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            tmp1 = np.clip(cat_image * 255.0, 0, 255.0)
            tmp2 = tmp1.astype('uint8')
            tmp2 = np.squeeze(tmp2)
            im = Image.fromarray(tmp2,mode = 'L')
            filepath = res_dir  + test_img_name
            im.save(filepath[:-4] + 'res.jpg')