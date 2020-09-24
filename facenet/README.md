## 制作镜像

* 制作 CPU 镜像

```
# docker build --compress -t facenet:1.0-cpu -f Dockerfile.cpu .

```

* 制作 GPU 镜像

```
# docker build --compress -t facenet:1.0-gpu -f Dockerfile.gpu .
```

## 训练

### CPU 训练示例

> CODE_PATH: facenet 目录存放路径（例如：/root/data/code）
> TRAIN_LOG_DIR：模型训练日志输出路径（例如：/root/data/log）
> DATASET_PATH：训练用数据集路径（例如：/root/dataset/casia_maxpy_mtcnnpy_182）
> LFW_DATASET_DIR：训练用验证数据集路径（例如：/root/dataset/lfw_mtcnnalign_160）
> LEARNING_RATE：学习率设置（例如：-1）
> MAX_NROF_EPOCHS：训练迭代次数（例如：1）
> EPOCH_SIZE：每次迭代 batch 的数量（例如：1）

```
# cp train-cpu.sh.example train-cpu.sh
# chmod a+x train-cpu.sh
# export CODE_PATH=<CODE_PATH>
# export TRAIN_LOG_DIR=<TRAIN_LOG_DIR>
# export DATASET_PATH=<DATASET_PATH>
# export LFW_DATASET_DIR=<LFW_DATASET_DIR>
# export LEARNING_RATE=<LEARNING_RATE>
# export MAX_NROF_EPOCHS=<MAX_NROF_EPOCHS>
# export EPOCH_SIZE=<EPOCH_SIZE>
# ./train-cpu.sh
```

### GPU 训练示例

> CODE_PATH: facenet 目录存放路径（例如：/root/data/code）
> TRAIN_LOG_DIR：模型训练日志输出路径（例如：/root/data/log）
> DATASET_PATH：训练用数据集路径（例如：/root/dataset/casia_maxpy_mtcnnpy_182）
> LFW_DATASET_DIR：训练用验证数据集路径（例如：/root/dataset/lfw_mtcnnalign_160）
> LEARNING_RATE：学习率设置（例如：-1）
> MAX_NROF_EPOCHS：训练迭代次数（例如：1）
> EPOCH_SIZE：每次迭代 batch 的数量（例如：1）

```
# cp train-gpu.sh.example train-gpu.sh
# chmod a+x train-cpu.sh
# export CODE_PATH=<CODE_PATH>
# export TRAIN_LOG_DIR=<TRAIN_LOG_DIR>
# export DATASET_PATH=<DATASET_PATH>
# export LFW_DATASET_DIR=<LFW_DATASET_DIR>
# export LEARNING_RATE=<LEARNING_RATE>
# export MAX_NROF_EPOCHS=<MAX_NROF_EPOCHS>
# export EPOCH_SIZE=<EPOCH_SIZE>
# ./train-gpu.sh
```

> 如果想要控制训练使用的GPU资源量，可以修改 train-gpu.sh 中的 --gpu_memory_fraction 指定使用的 GPU 资源的百分比


## 数据集裁剪

从源数据集(<SOURCE_DIR>)中，提取部分数据，保存到目标数据集(<DESTINATION_DIR>)中，制作成裁剪后的新的数据集

> 想要修改裁剪比例，可以修改脚本中的 CROP_PERCENT 变量，该变量控制 ***裁减掉*** 的百分比，默认裁剪掉 75%，即保留 25%（新的数据集包含源数据集的25%的数据）。

```
# crop-dataset_CASIA-maxpy-clean.sh <SOURCE_DIR> <DESTINATION_DIR>
```
