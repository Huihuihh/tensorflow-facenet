from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import random
from time import time

from os.path import join as pjoin
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
from openvino.inference_engine import IENetwork, IECore

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def main():      

    # Load the model 
    model_xml='lrmodels/facenet.xml'
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1
    n, c, h, w = net.inputs[input_blob].shape
    exec_net = ie.load_network(network=net, device_name='MYRIAD')

    image=[]
    nrof_images=0

    emb_dir='emb_img/'
    all_obj=[]
    for i in os.listdir(emb_dir):
        all_obj.append(i)
        img = misc.imread(os.path.join(emb_dir,i), mode='RGB')
        prewhitened = facenet.prewhiten(img)
        image.append(prewhitened)
        nrof_images=nrof_images+1

    images=np.stack(image)
    compare_embs = []
    for i in range(0, len(images)):
        known_image = images[i]
        known_image = known_image.reshape(1,3,160,160)
        compare = exec_net.infer(inputs={input_blob: known_image})[out_blob]
        compare_embs.append(compare)
    compare_emb=np.vstack(compare_embs)
    compare_num=len(compare_emb)

    #开启ip摄像头
    #video="http://admin:admin@192.168.0.107:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
    # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
    #capture =cv2.VideoCapture(video)
    #capture =cv2.VideoCapture('/home/awcloud/Desktop/huige.mp4')
    capture =cv2.VideoCapture(0)
    cv2.namedWindow("camera",1)
    k = 0
    count =3
    timer=0
    while True:
        ret, frame = capture.read() 

        if (k == 0):
            # rgb frame np.ndarray 480*640*3
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            # 获取 判断标识 bounding_box crop_image
            mark,crop_image,faces=load_and_align_data(rgb_frame,160,44)

            timer+=1
            if(1):
            
                print(timer)
                if(mark):
                    if crop_image is not None:
                        crop_image = crop_image.reshape(1,3,160,160)
                        detect_time = []
                        t0 =time()
                        emb = exec_net.infer(inputs={input_blob: crop_image})[out_blob]
                        detect_time.append((time() - t0) * 1000)
                        print('detect_time is {} ms'.format(np.average(np.asarray(detect_time))))
                        temp_num=len(emb)

                        fin_obj=[]
                        print(all_obj)

                        # 为bounding_box 匹配标签
                        for i in range(temp_num):
                            dist_list=[]
                            for j in range(compare_num):
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], compare_emb[j,:]))))
                                dist_list.append(dist)
                            min_value=min(dist_list)
                            print(dist_list)
                            if(min_value>0.22):
                                fin_obj.append('unknow')
                            else:
                                fin_obj.append(all_obj[dist_list.index(min_value)])    


                        # 在frame上绘制边框和文字
                        for (x,y,h,w) in faces:
                            for rec_position in range(temp_num):
                                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                                cv2.putText(frame,fin_obj[rec_position],(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), 2)

                    cv2.imshow('camera',frame)

            # cv2.imshow('camera',frame)
            key = cv2.waitKey(3)
            if key == 27:
                #esc键退出
                print("esc break...")
                break
        k = k + 1
        k = k % count

    # When everything is done, release the capture
    capture.release()
    cv2.destroyWindow("camera")
  
# 传入rgb np.ndarray
def load_and_align_data(img, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img_size = np.asarray(img.shape)[0:2]

    faces = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return 1, None, faces

    for (x,y,h,w) in faces:
        global cropped
        cropped = img[y:y+h, x:x+w]

    crop=[]
    aligned=misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    crop.append(prewhitened)

    # np.stack 将crop由一维list变为二维
    crop_image=np.stack(crop)  

    return 1,crop_image,faces


if __name__=='__main__':
    main()
