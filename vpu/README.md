#生成openvino需要的模型文件
`python /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo_tf.py --input_model models/facenet.pb --output_dir lrmodels/ --freeze_placeholder_with_value "phase_train->False"`

#使用openvino进行推理
默认使用vpu
`python face_recognition.py`


直接将想要测试的对象的一张照片以其英文名命名（中文会乱码），放入一个名为`test_img`文件夹中，接下来对其进行人脸检测并切割，切割后的人脸图片尺寸为160*160，存入`emb_img`文件夹中，这一步的主要目的是为了不要每次测试的时候都还要重新开始人脸检测，当人脸识别程序启动时，先读取`emb_img`文件夹图片并输入网络得到其emb（128维特征），用于后续与摄像头捕捉的照片进行比较


- 文件夹（涉及个人和同学照片，未上传，测试时自己直接新建即可）

  > test_img : 此文件夹中直接存放需要识别对象的一张照片
  >
  > emb_img: 此文件夹可以自己新建，或者不管（脚本中对这个文件夹检测了，没有则新建），用于存放剪切后的160*160尺寸的人脸图片

- .py文件（一个用来批处理图片，一个用来运行检测）

  > calculate_dection_face.py : 代码中已经注明了有些路径自己要更改一下，先执行此脚本，进行人脸定位切割（有点残忍的感觉）
  >
  > face_recognition.py : 直接执行即可，此次默认使用的是电脑自带的摄像头（如果要使用手机的，自己改一下，还是以前方法），路径也要注意

