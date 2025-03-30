import gradio as gr
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("config/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "checkpoints/model_final_280758.pkl"
    # 添加GPU配置
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg

class VisualizationDemo:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        outputs = self.predictor(image)
        v = Visualizer(image,
                      MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                      scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()

 # 初始化模型
cfg = setup_cfg()
faster_rcnn_demo = VisualizationDemo(cfg)

# 创建 Gradio 界面
with gr.Blocks() as gradio_demo:
    gr.Markdown("## 智能识别系统")

    with gr.Row():
        camera = gr.Image(source="webcam", streaming=True, label="摄像头预览")
        output_image = gr.Image(label="识别结果")

    # 开始识别
    start_btn = gr.Button("开始识别")
    start_btn.click(
        fn=faster_rcnn_demo.run_on_image,
        inputs=[camera],
        outputs=[output_image]
    )


if __name__ == "__main__":
    gradio_demo.launch(share=True)

