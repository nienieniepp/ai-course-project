# 基于深度学习的面部表情识别系统（FER-2013）

## 📌 项目简介
本项目基于 FER-2013 数据集，实现一个深度学习的人脸情绪识别系统，并通过 Flask 构建网页端展示界面。

系统可以：
- 对输入的人脸图像进行情绪分类
- 支持 7 类情绪识别：
  - Angry（愤怒）
  - Disgust（厌恶）
  - Fear（恐惧）
  - Happy（笑）
  - Sad（悲伤）
  - Surprise（惊讶）
  - Neutral（中性）
- 提供网页上传图片并实时预测结果

---

## 🎯 项目目标
- 构建一个基于深度学习的情绪识别模型
- 使用 Flask 实现可交互的网页应用
- 对模型进行训练、测试和评估
- 展示实验结果并进行分析

---

## 📂 数据集说明
- 数据集：FER-2013
- 图像尺寸：48 × 48 灰度图
- 训练集：28,709 张
- 测试集：3,589 张
- 分类类别：
  - 0：Angry（愤怒）
  - 1：Disgust（厌恶）
  - 2：Fear（恐惧）
  - 3：Happy（笑）
  - 4：Sad（悲伤）
  - 5：Surprise（惊讶）
  - 6：Neutral（中性）

---

## 🧠 方法说明
- 深度学习框架：PyTorch
- 模型：卷积神经网络（CNN）
- 损失函数：交叉熵损失（CrossEntropyLoss）
- 优化器：Adam

---

## 📁 项目结构
project/
│
├─ dataset/ # 数据集
├─ models/ # 模型权重
├─ results/ # 结果图（loss、accuracy等）
├─ static/ # 静态文件（CSS等）
├─ templates/ # HTML页面
├─ uploads/ # 用户上传图片
│
├─ app.py # Flask 主程序
├─ train.py # 模型训练
├─ predict.py # 单张图片预测
├─ model.py # 模型结构
├─ dataset.py # 数据加载
├─ utils.py # 工具函数
├─ requirements.txt # 依赖文件
└─ README.md # 项目说明


---

## ⚙️ 环境配置

```bash
pip install -r requirements.txt
🚀 模型训练
python train.py
训练完成后，模型将保存至：
models/emotion_cnn.pth
单张图片预测
python predict.py --image sample.jpg
输出内容：

预测情绪类别
各类别概率
🌐 启动网页系统
python app.py
浏览器访问：
http://127.0.0.1:5000
功能：

上传图片
显示预测结果
显示情绪概率分布
📊 实验结果

（运行训练后填写）

模型准确率：xx%
Loss 曲线
Accuracy 曲线
混淆矩阵（Confusion Matrix）
📌 结果分析
模型在 Happy、Neutral 类别上表现较好
在 Fear 与 Surprise 之间容易混淆
数据不平衡可能影响模型性能
🧾 结论

本项目成功实现了一个基于深度学习的情绪识别系统，并通过网页形式进行展示。

未来改进方向：

使用更复杂模型（如 ResNet、Transformer）
增加数据增强
实现实时摄像头识别
优化前端界面
