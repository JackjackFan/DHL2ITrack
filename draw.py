import cv2 as cv
import os 
img_list = sorted(os.listdir(os.path.join('/home/cv/data2/tracker_data/OTB100/Basketball/img')))
save_file = cv.VideoWriter((os.path.join('./','Basketball'+'.avi')),
    cv.VideoWriter_fourcc('M', 'J', 'P', "G"),25,(576,432))
indx = 0
for name in enumerate(img_list):
    name = name[1]
    file=open('/home/cv/data/fsh/TransT-main/results/OTB100/OTB100_Tracker_Results/OTB100/SiamFC/Basketball.txt')    
    gt=[]  
    for line in file.readlines():    
        curLine=line.strip().split(",")    
        floatLine=list(map(float,curLine))#这里使用的是map函数直接把数据转化成为float类型    
        gt.append(floatLine) 
    image = cv.imread(os.path.join('/home/cv/data2/tracker_data/OTB100/Basketball/img',name))
    res = cv.rectangle(image, (int(gt[indx][0]),int(gt[indx][1])),(int(gt[indx][0]+gt[indx][2]),int(gt[indx][1]+gt[indx][3])), (0,255,0),2)
    save_file.write(res)
    indx = indx+1
save_file.release()



