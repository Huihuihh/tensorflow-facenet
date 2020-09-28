import cv2

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def main(args):
    dection()
    
def dection():
    img_dir='test_img/'
    img_path_set=[]
    for file in os.listdir(img_dir):
        single_img=os.path.join(img_dir,file)
        print(single_img)
        print('loading...... :',file)
        img_path_set.append(single_img)

    images = load_and_align_data(img_path_set, 160, 44)

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

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

        for (x,y,h,w) in faces:
            cropped= img[y:y+h, x:x+w]

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
