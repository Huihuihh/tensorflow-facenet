from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import align.detect_face
import random

from os.path import join as pjoin
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
from openvino.inference_engine import IENetwork, IECore


def main():      

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 裁剪图片
            dection()
            # Load the model 
            # 这里要改为自己的模型位置
            model='/home/awcloud/Desktop/model/20180402-114759.pb'
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

            detect_image='/home/awcloud/Desktop/timg.jpg'

            frame = cv2.imread(detect_image) 
            # 获取 判断标识 bounding_box crop_image
            mark,bounding_box,crop_image=detect_load_and_align_data(detect_image,160,44)
            if(mark):
                feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
                temp_num=len(emb)

                fin_obj=[]

                # 为bounding_box 匹配标签
                for i in range(temp_num):
                    dist_list=[]
                    for j in range(compare_num):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[i], compare_emb[j]))))
                        dist_list.append(dist)
                    min_value=min(dist_list)
                    if(min_value>0.65):
                        fin_obj.append('unknow')
                    else:
                        fin_obj.append(all_obj[dist_list.index(min_value)])    


                # 在frame上绘制边框和文字
                for rec_position in range(temp_num):                        
                    cv2.rectangle(frame,(bounding_box[rec_position,0],bounding_box[rec_position,1]),(bounding_box[rec_position,2],bounding_box[rec_position,3]),(0, 255, 0), 2, 8, 0)

                    cv2.putText(
                        frame,
                    fin_obj[rec_position], 
                    (bounding_box[rec_position,0],bounding_box[rec_position,1]),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    0.8, 
                    (0, 0 ,255), 
                    thickness = 2, 
                    lineType = 2)

            # 保存一张图像
            cv2.imwrite("/home/awcloud/Desktop/detect_img/test.jpg",frame)
            

# 创建load_and_align_data网络
print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


# 传入rgb np.ndarray
def detect_load_and_align_data(img, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img = misc.imread(img, mode='RGB')
    img_size = np.asarray(img.shape)[0:2]

    # bounding_boxes shape:(1,5)  type:np.ndarray
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 如果未发现目标 直接返回
    if len(bounding_boxes) < 1:
        return 0,0,0

    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    #det = np.squeeze(bounding_boxes[:,0:4])
    det=bounding_boxes

    print('det shape type')
    print(det.shape)
    print(type(det))

    det[:,0] = np.maximum(det[:,0]-margin/2, 0)
    det[:,1] = np.maximum(det[:,1]-margin/2, 0)
    det[:,2] = np.minimum(det[:,2]+margin/2, img_size[1]-1)
    det[:,3] = np.minimum(det[:,3]+margin/2, img_size[0]-1)

    det=det.astype(int)
    crop=[]
    for i in range(len(bounding_boxes)):
        temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
        aligned=misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        crop.append(prewhitened)

    # np.stack 将crop由一维list变为二维
    crop_image=np.stack(crop)  

    return 1,det,crop_image

def dection():
    # 将目标图片文件夹下的图片地址append进list,传入load_and_align_data(),对图片进行切割（因为其图片参数为list）
    img_dir='test_img/'
    img_path_set=[]
    for file in os.listdir(img_dir):
        single_img=os.path.join(img_dir,file)
        print(single_img)
        print('loading...... :',file)
        img_path_set.append(single_img)

    images = cut_load_and_align_data(img_path_set, 160, 44)

    save_dir='save_img/'

    if(os.path.exists(save_dir)==False):
        os.mkdir(save_dir)

    count=0
    for file in os.listdir(img_dir):
        misc.imsave(os.path.join(save_dir,file),images[count])
        count=count+1

    emb_dir='emb_img/'

    if(os.path.exists(emb_dir)==False):
        os.mkdir(emb_dir)

    a = 0
    filename_list = os.listdir(save_dir)
    for i in filename_list:
        used_name = save_dir + filename_list[a]
        new_name = emb_dir + filename_list[a].strip().split('.')[0]
        os.rename(used_name,new_name)
        a += 1

    os.rmdir(save_dir)

def cut_load_and_align_data(image_paths, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        # img = misc.imread(image, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

        # 根据cropped位置对原图resize，并对新得的aligned进行白化预处理
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

if __name__=='__main__':
    main()
