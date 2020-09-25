# 训练

## 模型初体验

`python3 src/run_test.py`


## 模型训练

```python
python3 src/train_softmax.py \
--logs_base_dir /root/data/log \
--models_base_dir /root/data/output/model/ \
--data_dir /root/dataset/casia-maxpy-mtcnnpy/casia_maxpy_mtcnnpy_182/ \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir /root/dataset/lfw-mtcnnalign/lfw_mtcnnalign_160/ \
--optimizer ADAM \
--learning_rate -1 \
--max_nrof_epochs 1 \
--keep_probability 0.8 \
--random_crop \
--random_flip \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--prelogits_norm_loss_factor 5e-4 \
--epoch_size 1 \
--gpu_memory_fraction 0.8
```
