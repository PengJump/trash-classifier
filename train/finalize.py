# -*- coding: utf-8 -*-
"""从已保存的 trashnet.pt 评估并导出 ONNX + classes.json（无需重训）。

用法：python -u train/finalize.py
"""
import json
import torch
from torch.utils.data import DataLoader
import train as T


def main():
    model = T.ResNet18(num_classes=len(T.CLASSES))
    ckpt = torch.load(T.PT_PATH, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()

    items = T.build_items()
    n = len(items)
    n_tr = int(n * 0.7)
    n_va = int(n * 0.15)
    va_items = items[n_tr:n_tr + n_va]
    te_items = items[n_tr + n_va:]

    va_ld = DataLoader(T.TrashDataset(va_items, T.EvalTransform()),
                       batch_size=T.BATCH, shuffle=False, num_workers=2)
    te_ld = DataLoader(T.TrashDataset(te_items, T.EvalTransform()),
                       batch_size=T.BATCH, shuffle=False, num_workers=2)

    va_top1, va_f1 = T.evaluate(model, va_ld, "cpu")
    te_top1, te_f1 = T.evaluate(model, te_ld, "cpu")
    print(f"VAL  top1={va_top1:.4f}  macro_f1={va_f1:.4f}", flush=True)
    print(f"TEST top1={te_top1:.4f}  macro_f1={te_f1:.4f}", flush=True)

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, T.ONNX_PATH,
                      input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamic_axes=None, dynamo=False)
    with open(T.META_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": T.CLASSES, "mean": T.MEAN, "std": T.STD,
                   "input_size": 224,
                   "val_top1": va_top1, "val_macro_f1": va_f1,
                   "test_top1": te_top1, "test_macro_f1": te_f1},
                  f, ensure_ascii=False, indent=2)
    print("saved", T.ONNX_PATH, flush=True)
    print("saved", T.META_PATH, flush=True)


if __name__ == "__main__":
    main()
