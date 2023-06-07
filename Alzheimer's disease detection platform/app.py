import os.path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
# import tensorflow as tf
# from data_process import get_data, get_data_test
import onnxruntime
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


ort_session = onnxruntime.InferenceSession('static/resnext50_32x4d.onnx')
app = Flask(__name__)


# 平台的根目录
@app.route('/')
def index():  # put application's code here
    return render_template('index.html')


# 平台的检测页

@app.route('/upload', methods=['POST', 'GET'])
def show():  # put application's code here
    file = request.files['img']
    file_name = secure_filename(file.filename)

    upload_path = os.path.join('static/images', file_name)

    file.save(upload_path)
    result = predict(upload_path)
    return render_template('result.html', file_name=file_name, result=result)
    # render_template('index.html', file_name=file_name, result=result)

    # return '<script> alert(\"%s\");</script>' % result

# def predict(file_name):
#     model = tf.keras.models.load_model('static/model_AlexNet.tf')
#     X = get_data_test(file_name)
#     result = model.predict(X).argmax(axis=1)
#     return result


def predict(file_name):
    x = torch.randn(1, 3, 256, 256).numpy()
    x.shape
    # onnx runtime 输入
    ort_inputs = {'input': x}

    # onnx runtime 输出
    ort_output = ort_session.run(['output'], ort_inputs)[0]
    ort_output.shape
    # 测试集图像预处理-RCTN：缩放裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])
    img_path = file_name
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil)
    input_img.shape
    input_tensor = input_img.unsqueeze(0).numpy()
    input_tensor.shape
    # ONNX Runtime 输入
    ort_inputs = {'input': input_tensor}
    # ONNX Runtime 输出
    pred_logits = ort_session.run(['output'], ort_inputs)[0]
    pred_logits = torch.tensor(pred_logits)
    pred_logits.shape
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
    pred_softmax.shape
    # 取置信度最大的 n 个结果
    n = 1
    top_n = torch.topk(pred_softmax, n)
    top_n
    # 预测类别
    pred_ids = top_n.indices.numpy()[0]
    pred_ids
    # 预测置信度
    confs = top_n.values.numpy()[0]
    confs
    # 载入类别和对应 ID
    idx_to_labels = np.load('static/idx_to_labels.npy', allow_pickle=True).item()
    # for i in range(n):
    class_name = idx_to_labels[pred_ids[0]]  # 获取类别名称
    confidence = confs[0] * 100  # 获取置信度
    # text = '{:<6} {:>.3f}'.format(class_name, confidence)
    text = class_name
    if text=='CN':
        text="健康"
    else :
        if text == 'CI':
            text='轻度认知障碍'
        else:
            text='阿尔兹海默症'


    return text
    # return pred_ids[0]

if __name__ == '__main__':
    app.run()
