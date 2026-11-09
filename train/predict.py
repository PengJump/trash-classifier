# -*- coding: utf-8 -*-
"""用 onnxruntime 快速验证 trashnet.onnx，命令行传入一张图即可。
用法：python predict.py <图片路径> [模型路径] [classes.json]
"""
import sys
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    nw, nh = round(w * s), round(h * s)
    img = img.resize((nw, nh), Image.BILINEAR)
    x0, y0 = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((x0, y0, x0 + 224, y0 + 224))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(MEAN)) / np.array(STD)
    a = a.transpose(2, 0, 1)[None]  # 1,3,224,224
    return a.astype(np.float32)


def main():
    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "trashnet.onnx"
    meta = sys.argv[3] if len(sys.argv) > 3 else "classes.json"

    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    classes = m["classes"]

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    x = preprocess(path)
    logits = sess.run(["logits"], {"input": x})[0][0]
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    order = np.argsort(-p)

    print("top3:")
    for i in order[:3]:
        print(f"  {classes[i]:10s} {p[i] * 100:5.1f}%")
    top1, top2 = order[0], order[1]
    if p[top1] < 0.7 or (p[top1] - p[top2]) < 0.15:
        print("-> 不确定（低于阈值或 top1/top2 太接近）")
    else:
        print(f"-> {classes[top1]}  {p[top1] * 100:.1f}%")


if __name__ == "__main__":
    main()
