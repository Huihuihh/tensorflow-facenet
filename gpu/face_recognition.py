from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import random

from os.path import join as pjoin
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from time import time


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
from openvino.inference_engine import IENetwork, IECore

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def main():      

    with tf.Graph().as_default():
        with tf.Session() as sess:     
            # Load the model 
            model='models/facenet.pb'
            model_exp = os.path.expanduser(model)
            with gfile.FastGFile(model_exp,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image=[]
            nrof_images=0

            # 这里要改为自己emb_img文件夹的位置
            emb_dir='emb_img/'
            all_obj=[]
            for i in os.listdir(emb_dir):
                all_obj.append(i)
                img = misc.imread(os.path.join(emb_dir,i), mode='RGB')
                prewhitened = prewhiten(img)
                image.append(prewhitened)
                nrof_images=nrof_images+1

            images=np.stack(image)
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            compare_emb = sess.run(embeddings, feed_dict=feed_dict) 
            compare_num=len(compare_emb)

            #开启ip摄像头
            #video="http://admin:admin@192.168.0.107:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
            # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
            #capture =cv2.VideoCapture(video)
            capture =cv2.VideoCapture(0)
            cv2.namedWindow("camera",1)
            k = 0
            count = 3
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
                                feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }
                                detect_time = []
                                t0 = time()
                                emb = sess.run(embeddings, feed_dict=feed_dict)
                                detect_time.append((time() - t0) * 1000)
                                print('detect_time is {} ms'.format(np.average(np.asarray(detect_time))))
                                temp_num=len(emb)

                                fin_obj=[]
                                print(all_obj)

                                # 为bounding_box 匹配标签
                                for i in range(temp_num):
                                    dist_list=[]
                                    for j in range(compare_num):
                                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[i], compare_emb[j]))))
                                        dist_list.append(dist)
                                    min_value=min(dist_list)
                                    print(dist_list)
                                    if(min_value>0.65):
                                        fin_obj.append('unknow')
                                    else:
                                        fin_obj.append(all_obj[dist_list.index(min_value)])    

                                for (x,y,h,w) in faces:
                                    for rec_position in range(temp_num):
                                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                                        cv2.putText(frame,fin_obj[rec_position],(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (128, 128, 0), 2)

                            cv2.imshow('camera',frame)


                    key = cv2.waitKey(3)
                    if key == 27:
                        #esc键退出
                        print("esc break...")
                        break

                k = k + 1
                k = k % count
            capture.release()
            cv2.destroyWindow("camera")

# 创建load_and_align_data网络
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# 传入rgb np.ndarray
def load_and_align_data(img, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    faces = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return 1, None, faces

    for (x,y,h,w) in faces:
        global cropped
        cropped = img[y:y+h, x:x+w]

    crop=[]
    aligned=misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = prewhiten(aligned)
    crop.append(prewhitened)

    # np.stack 将crop由一维list变为二维
    crop_image=np.stack(crop)  

    return 1,crop_image,faces

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

if __name__=='__main__':
    main()
