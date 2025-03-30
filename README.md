# 📖操作手册

## 训练命令(VOC2007)
**训练集环境变量设置**
```
export DETECTRON2_DATASETS=/home/zhaoxuekun/my_project/datasets/VOCdevkit
```
**执行训练**
```
python train_faster_rcnn.py --num-gpus 2 \
                            --config-file config/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
```

**启动gradio**
```
python gradio_demo.py
```

**infer**

以COCO的faster_rcnn_R_50_FPN为例
```
python faster_rcnn.py --config-file config/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml \
                      --input ./JPEGImages/000001.jpg \
                      --opts MODEL.WEIGHTS ./checkpoints/model_final_280758.pkl
```