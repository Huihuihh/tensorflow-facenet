# 生成openvino需要的模型文件
`python /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo_tf.py --input_model models/facenet.pb --output_dir lrmodels/ --freeze_placeholder_with_value "phase_train->False"`

# 使用openvino进行推理
默认使用vpu
`python face_recognition.py`
