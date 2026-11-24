# -*- coding: utf-8 -*-
"""
YOLO26 训练脚本 —— MetalScan 四分类缺陷检测
使用 notebook 内核环境 (D:/Apps/yolo/gpu_venv, RTX 5060 CUDA) 训练。

用法 (在该内核环境的 python 下运行):
  python train_yolo26.py                # 默认 100 epochs
  python train_yolo26.py --epochs 60 --imgsz 640 --batch 16

训练完成后自动:
  1) 在验证集上评估 (mAP 指标)
  2) 导出 ONNX 到 runs/metalscan_yolo26/weights/best.onnx
  3) 用 best.pt 对验证集预测并保存可视化图到 val_predictions\
"""
import argparse
import glob
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
PROJECT = os.path.join(BASE_DIR, "runs")
NAME = "metalscan_yolo26"


def find_weights():
    """在 runs/ 下查找最近一次训练的 best.pt。"""
    cands = sorted(glob.glob(os.path.join(PROJECT, NAME, "weights", "best.pt")))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolo26n.pt", help="预训练模型: yolo26n.pt / yolo26s.pt / yolov8n.pt")
    ap.add_argument("--device", default=None, help="0=GPU, cpu; 默认自动")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--skip-vis", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    device = args.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")

    model = YOLO(args.model)  # 自动下载预训练权重
    results = model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        patience=30,
        workers=4,
        seed=42,
    )

    best = find_weights()
    if not best:
        print("[错误] 未找到训练产物 best.pt"); sys.exit(1)
    print(f"[train] 最佳权重: {best}")

    # ---- 验证集评估 ----
    best_model = YOLO(best)
    metrics = best_model.val(data=DATA_YAML, imgsz=args.imgsz, device=device)
    print("=" * 60)
    print(f"[val] mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    print(f"[val] 各类 mAP50:", {best_model.names[k]: round(v, 3) for k, v in zip(metrics.box.ap_class_index, metrics.box.ap50)})

    # ---- ONNX 导出 ----
    if not args.skip_onnx:
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=13)
        print(f"[onnx] 已导出: {onnx_path}")
        # 复制到 weights 目录旁
        dst = os.path.join(os.path.dirname(best), os.path.basename(onnx_path))
        if os.path.abspath(onnx_path) != os.path.abspath(dst):
            shutil.copyfile(onnx_path, dst)
            print(f"[onnx] 已复制: {dst}")

    # ---- 验证集预测可视化 ----
    if not args.skip_vis:
        vis_dir = os.path.join(BASE_DIR, "val_predictions")
        if os.path.isdir(vis_dir):
            shutil.rmtree(vis_dir)
        val_imgs = sorted(glob.glob(os.path.join(BASE_DIR, "dataset", "images", "val", "*.jpg")))
        print(f"[vis] 对验证集 {len(val_imgs)} 张图片预测并保存可视化...")
        best_model.predict(
            val_imgs, conf=0.45, save=True, project=BASE_DIR, name="val_predictions", exist_ok=True
        )
        print(f"[vis] 可视化已保存: {os.path.join(BASE_DIR, 'val_predictions')}")

    print("=" * 60)
    print("全部完成! 应用可直接使用:")
    print(f"  python metal_scan_app.py --camera           # 摄像头检测")
    print(f"  python metal_scan_app.py --folder images    # 批量测试 images\\ 下图片")


if __name__ == "__main__":
    main()
