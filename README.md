# Trash Classifier

基于图像的垃圾分类识别：上传一张物品照片，识别材质类别，并给出投放提示；识别没有把握时返回「不确定」。

## 功能

- 识别 6 类材质：纸箱（cardboard）、玻璃（glass）、金属（metal）、纸类（paper）、塑料（plastic）、其他垃圾（trash）
- 置信度不足时自动拒识，返回「不确定」，支持手动纠正
- 识别结果附带投放提示：可回收物 / 其他垃圾，以及有害垃圾、脏污残留等兜底文案
- 后端内置 Web 界面，无需单独部署前端

## 环境要求

- 运行：Java 17 或更高版本
- 构建后端（仅首次需要）：Java 17 + Maven 3.6+

## 快速开始

### 1. 构建后端（首次运行前执行一次）

仓库不提交编译产物（`backend/trash-classifier.jar` 约 103MB，超出 GitHub 单文件 100MB 上限），首次运行前先构建：

```powershell
cd backend
mvn package
copy target\trash-classifier-0.0.1-SNAPSHOT.jar trash-classifier.jar
```

`mvn package` 会把前端页面和 ONNX Runtime 一起打进 jar（首次会从 Maven Central 下载依赖，约 90MB）；模型 `trashnet.onnx` 不打包，运行时从项目根目录加载。

### 2. 一键启动

1. 双击 `start-backend.bat`
2. 浏览器自动打开 <http://localhost:8080>
3. 上传 `samples/` 里的任意一张图片，点击「识别」

### 3. 手动启动

```powershell
cd backend
java -jar trash-classifier.jar --trash.model.path=..\trashnet.onnx
```

打开 <http://localhost:8080>。

模型路径用 `--trash.model.path=` 指定，默认使用当前目录下的 `trashnet.onnx`。

## API

### GET /api/health

返回服务状态与类别列表：`{"ok":true,"classes":[...]}`。

### POST /api/classify

multipart 字段 `file` 上传图片，返回：

```json
{
  "label": "plastic",
  "labelCn": "塑料",
  "confidence": 0.998,
  "category": "可回收物",
  "tip": "塑料 · 倒空、简单冲洗",
  "probs": { "plastic": 0.998, "glass": 0.001 }
}
```

命令行自测：

```powershell
curl -F "file=@samples/plastic.jpg" http://localhost:8080/api/classify
```

## 模型

- 网络：ResNet18，ImageNet 预训练 + TrashNet 微调，输入 224×224
- 类别：cardboard / glass / metal / paper / plastic / trash
- 归一化：ImageNet mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]
- 拒识：`top1 置信度 < 0.70` 或 `top1 - top2 < 0.15` 时返回「不确定」

### 投放规则

| 材质类别 | 投放建议 |
| --- | --- |
| 纸箱 / 玻璃 / 金属 / 纸类 / 塑料 | 可回收物 |
| 其他垃圾 | 其他垃圾 |

兜底提示（界面文案，非模型分类）：电池 / 药品 / 灯管 → 有害垃圾；脏污 / 有残留 → 其他垃圾。

## 性能

TrashNet 数据集，70/15/15 划分（seed 42）：

| 指标 | 验证集 | 测试集 |
| --- | --- | --- |
| Top-1 准确率 | 88.1% | 86.6% |
| Macro-F1（6 类平均） | 86.7% | 82.7% |

## 重新训练

项目自带训练好的模型（`trashnet.onnx`）与 PyTorch 权重（`trashnet.pt`），默认无需重训。如需复现或更换数据：

```powershell
pip install torch numpy onnx onnxruntime ml-dtypes

python -u train/train.py     # 训练（10 epoch，CPU 约 7 分钟/epoch）
python -u train/finalize.py  # 评估并导出 trashnet.onnx + classes.json
```

- `train.py` 不依赖 torchvision，数据增强与 ResNet18 结构均为手写实现
- 预训练权重 `weights/resnet18-f37072fd.pth` 只加载主干，跳过 1000 类的 fc 层
- 训练脚本路径均相对脚本自身位置解析，整个目录可整体拷贝到其他机器使用

## 技术栈

- 后端：Spring Boot 3 + ONNX Runtime（Java）
- 前端：Vue 3 + Vite
- 训练：PyTorch，导出 ONNX

## 目录结构

```
.
├── start-backend.bat        # 一键启动脚本
├── trashnet.onnx            # 训练好的模型（推理用）
├── trashnet.pt              # PyTorch 最优权重（重训用）
├── classes.json             # 类别与指标
├── samples/                 # 示例图片（每类一张）
├── weights/                 # ResNet18 预训练权重
├── data/dataset-resized/    # TrashNet 数据集（6 类 2527 张）
├── backend/                 # Spring Boot 后端源码与打包产物
├── frontend/                # Vue 3 前端源码
└── train/                   # 训练 / 导出 / 单图预测脚本
```

## 数据来源

公开数据集 [TrashNet](https://github.com/garythung/trashnet)，6 类共 2527 张，无需自采或人工标注。
