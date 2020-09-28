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

#import matplotlib
#matplotlib.use('Agg')
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def main():      

    # Load the model 
    # 这里要改为自己的模型位置
    #model='/home/awcloud/Desktop/model/'
    #facenet.load_model(model)
    #model_xml='/home/awcloud/Desktop/model_output/20180402-114759.xml'
    model_xml='/home/awcloud/Desktop/data_output/model-20200713-181635.xml'
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    #input_blob,out_blob,net.batch_size = next(iter(net.inputs)),next(iter(net.outputs)),1
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1
    n, c, h, w = net.inputs[input_blob].shape
    exec_net = ie.load_network(network=net, device_name='MYRIAD')
    #exec_net = ie.load_network(network=net, device_name='CPU')

    # Get input and output tensors
    #images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


    image=[]
    nrof_images=0

    # 这里要改为自己emb_img文件夹的位置
    emb_dir='/home/awcloud/Desktop/emb_img'
    all_obj=[]
    for i in os.listdir(emb_dir):
        all_obj.append(i)
        img = misc.imread(os.path.join(emb_dir,i), mode='RGB')
        prewhitened = facenet.prewhiten(img)
        image.append(prewhitened)
        nrof_images=nrof_images+1

    #print(image[0])
    #print(image[1])
    #print(image[2])
    images=np.stack(image)
    compare_embs = []
    for i in range(0, len(images)):
        known_image = images[i]
        #known_image = np.ndarray(shape=(n, c, h, w))
        #known_image = cv2.resize(known_image, (w, h))
        #known_image = known_image.transpose((2, 0, 1))
        known_image = known_image.reshape(1,3,160,160)
        compare = exec_net.infer(inputs={input_blob: known_image})[out_blob]
        #print(compare)
        #print(i)
        #print(np.array(compare).shape)
        compare_embs.append(compare)
    #compare_emb=np.vstack((compare_embs[0],compare_embs[1],compare_embs[2],compare_embs[3],compare_embs[4]))
    compare_emb=np.vstack(compare_embs)
    #feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    #compare_emb = sess.run(embeddings, feed_dict=feed_dict) 
    compare_num=len(compare_emb)

    #开启ip摄像头
    #video="http://admin:admin@192.168.0.107:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
    # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
    #capture =cv2.VideoCapture(video)
    capture =cv2.VideoCapture('/home/awcloud/Desktop/huige.mp4')
    #capture =cv2.VideoCapture(0)
    cv2.namedWindow("camera",1)
    k = 0
    count =3
    timer=0
    while True:
        all_time = []
        t0 = time()
        ret, frame = capture.read() 

        if (k == 0):
            # rgb frame np.ndarray 480*640*3
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            # 获取 判断标识 bounding_box crop_image
            cut_time = []
            t1 = time()
            mark,crop_image,faces=load_and_align_data(rgb_frame,160,44)
            cut_time.append((time() - t1) * 1000)
            print('cut_time is {} ms'.format(np.average(np.asarray(cut_time))))

            timer+=1
            if(1):
            
                print(timer)
                if(mark):
                    if crop_image is not None:
                        crop_image = crop_image.reshape(1,3,160,160)
                        detect_time = []
                        t2 =time()
                        emb = exec_net.infer(inputs={input_blob: crop_image})[out_blob]
                        detect_time.append((time() - t2) * 1000)
                        print('detect_time is {} ms'.format(np.average(np.asarray(detect_time))))
                        temp_num=len(emb)

                        fin_obj=[]
                        print(all_obj)

                        # 为bounding_box 匹配标签
                        for i in range(temp_num):
                            dist_list=[]
                            for j in range(compare_num):
                                #dist = distance.cosine(emb[i,:], compare_emb[j,:])
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], compare_emb[j,:]))))
                                dist_list.append(dist)
                            min_value=min(dist_list)
                            print(dist_list)
                            if(min_value>0.20):
                                fin_obj.append('unknow')
                            else:
                                fin_obj.append(all_obj[dist_list.index(min_value)])    


                        # 在frame上绘制边框和文字
                        #faces = faceCascade.detectMultiScale(rgb_frame, scaleFactor=1.2, minNeighbors=5)
                        for (x,y,h,w) in faces:
                            for rec_position in range(temp_num):
                                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                                #roi_gray = rgb_frame[y:y+h, x:x+w]
                                #roi_color = frame[y:y+h, x:x+w]
                                cv2.putText(frame,fin_obj[rec_position],(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), 2)


                    cv2.imshow('camera',frame)

                    all_time.append((time() - t0) * 1000)
                    print('all_time is {} ms'.format(np.average(np.asarray(all_time))))

            # cv2.imshow('camera',frame)
            key = cv2.waitKey(3)
            if key == 27:
                #esc键退出
                print("esc break...")
                break
        k = k + 1
        k = k % count

    # if key == ord(' '):
    #     # 保存一张图像
    #     num = num+1
    #     filename = "frames_%s.jpg" % num
    #     cv2.imwrite(filename,frame)
    
    # When everything is done, release the capture
    capture.release()
    cv2.destroyWindow("camera")
  


        # When everything is done, release the capture


# 创建load_and_align_data网络
print('Creating networks and loading parameters')
#with tf.Graph().as_default():
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
#    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#    with sess.as_default():
#        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


# 修改版load_and_align_data
# 传入rgb np.ndarray
def load_and_align_data(img, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img_size = np.asarray(img.shape)[0:2]

    # bounding_boxes shape:(1,5)  type:np.ndarray
    #bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return 1, None, faces

    for (x,y,h,w) in faces:
        global cropped
        cropped = img[y:y+h, x:x+w]

    # 如果未发现目标 直接返回
    #if len(bounding_boxes) < 1:
    #    return 0,0,0

    ## 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    ##det = np.squeeze(bounding_boxes[:,0:4])
    #det=bounding_boxes

    #print('det shape type')
    #print(det.shape)
    #print(type(det))

    #det[:,0] = np.maximum(det[:,0]-margin/2, 0)
    #det[:,1] = np.maximum(det[:,1]-margin/2, 0)
    #det[:,2] = np.minimum(det[:,2]+margin/2, img_size[1]-1)
    #det[:,3] = np.minimum(det[:,3]+margin/2, img_size[0]-1)

    #det=det.astype(int)
    crop=[]
    #for i in range(len(bounding_boxes)):
    #    temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
    aligned=misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = facenet.prewhiten(aligned)
    crop.append(prewhitened)

    # np.stack 将crop由一维list变为二维
    crop_image=np.stack(crop)  

    return 1,crop_image,faces
    #return 1,det,crop



if __name__=='__main__':
    main()
