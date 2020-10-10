from time import time
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import random

from tkinter import *
from PIL import Image, ImageTk
from tensorflow.python.platform import gfile

import sklearn

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def main():      
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    sess = tf.Session(config=soft_config)
    sess.run(tf.global_variables_initializer())
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

    #开启ip摄像头
    #video="http://admin:admin@192.168.0.107:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
    # 参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
    #capture =cv2.VideoCapture(video)
    capture =cv2.VideoCapture(0)
    cv2.namedWindow("camera",1)
    #cv2.resizeWindow("camera", 50, 10)
    k = 0
    count = 3
    timer=0
    while True:
        ret, frame = capture.read() 
        if (k == 0):
            # rgb frame np.ndarray 480*640*3
            rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
            # 获取 判断标识 bounding_box crop_image
            mark,crop_image,faces=load_and_align_data(rgb_frame,160)

            img_dir='test_img/'
            if(os.path.exists(img_dir)==False):
                os.mkdir(img_dir)

            key = cv2.waitKey(3)
            if key == 119:
                cv2.imwrite("test_img/test.jpg", frame)
                APP()
                # 裁剪图片
                dection()

            emb_dir='emb_img/'
            if(os.path.exists(emb_dir)==False):
                os.mkdir(emb_dir)

            if not os.listdir(emb_dir):
                for (x,y,h,w) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.putText(frame,'unknow',(x,y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (128, 128, 0), 2)
                cv2.imshow('camera',frame)
                if key == 27:
                    print("esc break...")
                    break
            else:
                image=[]
                nrof_images=0
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
                                
                timer+=1
                if(1):
                    print(timer)
                    if(mark):
                        if crop_image is not None:
                            feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }
                            detect_time = []
                            t2 = time()
                            emb = sess.run(embeddings, feed_dict=feed_dict)
                            detect_time.append((time() - t2) * 1000)
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
                                    cv2.putText(frame,fin_obj[rec_position],
                                            (x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (128, 128, 0), 2)
                        cv2.imshow('camera',frame)
                if key == 27:
                    print("esc break...")
                    break

        k = k + 1
        k = k % count

    # When everything is done, release the capture
    capture.release()
    cv2.destroyWindow("camera")
  
# 传入rgb np.ndarray
def load_and_align_data(img, image_size):
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

class APP:
    def __init__(self):
        self.root = Tk()
        self.root.title('FACE')
        self.root.geometry('%dx%d' % (800, 800))
        self.createFirstPage()
        mainloop()

    def createFirstPage(self):
        self.page1 = Frame(self.root)
        self.page1.pack()
        w_box = 800
        h_box = 600
        image = Image.open("test_img/test.jpg") #随便使用一张图片 不要太大
        w, h = image.size
        image_resized = self.resize(w, h, w_box, h_box, image)
        photo = ImageTk.PhotoImage(image = image_resized)
        self.data1 = Label(self.page1, width=w_box, height=h_box, image=photo)
        self.data1.image = photo
        self.data1.pack(padx=5, pady=5)
        self.entry = Entry(self.page1,bd=4)
        self.entry.pack()
        self.button11 = Button(self.page1, width=18, height=2, text="确认", bg='red', 
                font=("宋", 12), relief='raise',command = self.quitMain)
        self.button11.pack(padx=25, pady = 10)

    def quitMain(self):
        name = self.entry.get()
        new_name = "test_img/" + name + '.jpg'
        os.rename("test_img/test.jpg",new_name)
        self.root.destroy()

    def resize(self, w, h, w_box, h_box, pil_image):  
        f1 = 1.0*w_box/w # 1.0 forces float division in Python2  
        f2 = 1.0*h_box/h  
        factor = min([f1, f2])  
        width = int(w*factor)  
        height = int(h*factor)  
        return pil_image.resize((width, height), Image.ANTIALIAS)


def dection():
    img_dir='test_img/'
    img_path_set=[]
    for file in os.listdir(img_dir):
        single_img=os.path.join(img_dir,file)
        print(single_img)
        print('loading...... :',file)
        img_path_set.append(single_img)

    images = cut_load_and_align_data(img_path_set, 160)

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

def cut_load_and_align_data(image_paths, image_size):
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

        for (x,y,h,w) in faces:
            global cropped
            cropped= img[y:y+h, x:x+w]

        # 根据cropped位置对原图resize，并对新得的aligned进行白化预处理
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

if __name__=='__main__':
    main()
