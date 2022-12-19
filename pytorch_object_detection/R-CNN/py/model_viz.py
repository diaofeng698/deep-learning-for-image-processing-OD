# sudo apt-get install graphviz
# pip install torchviz
# 模型结构可视化
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
import torchvision.models as models

model = models.alexnet(pretrained=True)
# print(model)
num_features = model.classifier[6].in_features
# 把alexnet变成二分类
model.classifier[6] = nn.Linear(num_features, 2)
print(model)

x = torch.randn(1, 3, 227, 227).requires_grad_(True)
y = model(x)

vis_graph = make_dot(y, params=dict(model.named_parameters()))
# 指定文件生成的文件夹
vis_graph.directory="/data/fdiao/learning/deep-learning-for-image-processing-OD/pytorch_object_detection/R-CNN/py"
# 指定文件生成的格式
vis_graph.format="png"
vis_graph.render("model_viz", view=True)