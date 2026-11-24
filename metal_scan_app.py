# -*- coding: utf-8 -*-
"""
MetalScan 工业零件表面缺陷检测系统
====================================
四分类: 0=scratch(划痕) 1=dent(缺角/凹陷) 2=stain(脏污/锈斑) 3=normal_part(正常件)

用法:
  # 1) 摄像头实时检测（默认 0 号摄像头，按 Q/ESC 退出，按 S 抓拍保存）
  python metal_scan_app.py --camera
  python metal_scan_app.py --camera --camera-id 1 --conf 0.4

  # 2) 单张图片检测
  python metal_scan_app.py --image path/to/part.jpg

  # 3) 文件夹批量检测（默认使用 images\ 下的测试图片）
  python metal_scan_app.py --folder images
  python metal_scan_app.py --folder images --save results

模型优先级: ONNX(部署模式) > 训练产物 best.pt > 初始权重
"""
import argparse
import os
import sys
import time
from datetime import datetime

import cv2

# 禁止 ultralytics 运行时自动 pip 安装（离线模式，避免部署环境联网依赖）
os.environ.setdefault("YOLO_OFFLINE", "1")

# Windows 控制台中文输出保护（GBK 终端下防 UnicodeEncodeError）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------- 配置 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PT = os.path.join(BASE_DIR, "runs", "metalscan_yolo26", "weights", "best.pt")
DEFAULT_MODEL_ONNX = os.path.join(BASE_DIR, "runs", "metalscan_yolo26", "weights", "best.onnx")
FALLBACK_MODEL = os.path.join(BASE_DIR, "weights", "yolo26n.pt")

CLASS_NAMES = {0: "scratch 划痕", 1: "dent 缺角", 2: "stain 脏污", 3: "normal_part 正常件"}
NG_CLASSES = {0, 1, 2}          # 检出即 NG
COLORS = {0: (0, 0, 255), 1: (0, 90, 255), 2: (140, 0, 255), 3: (0, 200, 0)}


def resolve_model(user_path=None, prefer_onnx=True):
    """按优先级解析模型文件。"""
    candidates = []
    if user_path:
        candidates.append(user_path)
    if prefer_onnx:
        candidates.append(DEFAULT_MODEL_ONNX)
    candidates += [DEFAULT_MODEL_PT, FALLBACK_MODEL]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "未找到模型文件，请先训练: python train_yolo26.py\n"
        "期望位置: %s" % DEFAULT_MODEL_PT
    )


def build_model(model_path, conf=0.5):
    from ultralytics import YOLO
    model = YOLO(model_path)
    # ONNX 用 onnxruntime 推理，PT 用原生
    return model


def draw_result(frame, result, elapsed_ms, model_tag):
    """绘制判定结果：左上角大字 OK/NG + 缺陷框 + 中文标签。"""
    names = result.names if hasattr(result, "names") else {}
    boxes = result.boxes
    detected = []
    for b in boxes:
        cls_id = int(b.cls[0])
        cn = names.get(cls_id, str(cls_id))
        conf = float(b.conf[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        color = COLORS.get(cls_id, (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cn} {conf:.2f}"
        # 标签防出画
        ty = y1 - 8 if y1 - 24 > 0 else y2 + 20
        cv2.rectangle(frame, (x1, ty - 18), (x1 + 8 * len(label) + 6, ty + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        detected.append((cls_id, cn, conf))

    is_ng = any(c in NG_CLASSES for c, _, _ in detected)
    verdict = "NG" if is_ng else "OK"
    color = (0, 0, 255) if is_ng else (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (190, 64), color, -1)
    cv2.putText(frame, verdict, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4, cv2.LINE_AA)

    # 状态栏
    bar = f"{model_tag} | {elapsed_ms:.0f}ms | defects:{len(detected)}"
    cv2.putText(frame, bar, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, verdict, detected


# ---------------- 摄像头模式 ----------------
def run_camera(args):
    model_path = resolve_model(args.model, prefer_onnx=not args.no_onnx)
    model_tag = os.path.basename(model_path)
    print(f"[MetalScan] 模型: {model_path}")
    model = build_model(model_path, conf=args.conf)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {args.camera_id}，请检查设备/占用。")
        sys.exit(1)
    print("[MetalScan] 摄像头已打开 | Q/ESC 退出 | S 抓拍保存到 results\\")

    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    win = "MetalScan Industrial Inspection - Q quit / S snapshot"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[警告] 读取摄像头失败，重试中...")
            time.sleep(0.3)
            continue
        t0 = time.time()
        result = model(frame, conf=args.conf, verbose=False)[0]
        elapsed = (time.time() - t0) * 1000
        frame, verdict, detected = draw_result(frame, result, elapsed, model_tag)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            fn = os.path.join(BASE_DIR, "results", f"cam_{datetime.now():%Y%m%d_%H%M%S}_{verdict}.jpg")
            cv2.imwrite(fn, frame)
            print("[抓拍]", fn, "->", verdict, [d[1] for d in detected])
    cap.release()
    cv2.destroyAllWindows()


# ---------------- 图片模式 ----------------
def run_images(args):
    model_path = resolve_model(args.model, prefer_onnx=not args.no_onnx)
    model_tag = os.path.basename(model_path)
    print(f"[MetalScan] 模型: {model_path}")
    model = build_model(model_path, conf=args.conf)

    # 收集待测图片
    if args.image:
        img_paths = [args.image]
    else:
        folder = args.folder or os.path.join(BASE_DIR, "images")
        img_paths = []
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_paths.append(os.path.join(root, f))
        if args.limit:
            img_paths = img_paths[: args.limit]
    if not img_paths:
        print("[错误] 未找到测试图片。")
        sys.exit(1)

    out_dir = os.path.join(BASE_DIR, args.save or "results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[MetalScan] 待测图片 {len(img_paths)} 张 -> 输出 {out_dir}")

    stats = {"OK": 0, "NG": 0}
    t_start = time.time()
    for i, p in enumerate(img_paths, 1):
        frame = cv2.imread(p)
        if frame is None:
            print(f"  [跳过] 无法读取: {p}")
            continue
        t0 = time.time()
        result = model(frame, conf=args.conf, verbose=False)[0]
        elapsed = (time.time() - t0) * 1000
        frame, verdict, detected = draw_result(frame, result, elapsed, model_tag)
        stats[verdict] += 1

        stem = os.path.splitext(os.path.basename(p))[0]
        cls_hint = os.path.basename(os.path.dirname(p))  # 图片所在类文件夹名（真实标签提示）
        out_name = f"{i:03d}_{stem}_pred{verdict}_true-{cls_hint}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), frame)
        marks = [f"{cn}({cf:.2f})" for _, cn, cf in detected]
        print(f"  [{i:3d}/{len(img_paths)}] {os.path.basename(p):40s} 判定={verdict:2s} 检出={marks or '无'} {elapsed:.0f}ms")

    cost = time.time() - t_start
    print("-" * 70)
    print(f"完成: {stats['OK']+stats['NG']} 张 | OK {stats['OK']} / NG {stats['NG']} | 总耗时 {cost:.1f}s "
          f"| 平均 {cost/(stats['OK']+stats['NG'])*1000:.0f}ms/张")
    print(f"可视化结果已保存: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="MetalScan 工业零件表面缺陷检测系统")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", action="store_true", help="摄像头实时检测")
    g.add_argument("--image", help="单张图片检测")
    g.add_argument("--folder", help="文件夹批量检测（默认 images\\）")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.45, help="置信度阈值")
    ap.add_argument("--model", help="指定模型路径（默认自动解析）")
    ap.add_argument("--save", default="results", help="结果保存目录")
    ap.add_argument("--limit", type=int, help="批量模式最多测 N 张")
    ap.add_argument("--no-onnx", action="store_true", help="不优先使用 ONNX，直接用 best.pt")
    args = ap.parse_args()

    if args.camera:
        run_camera(args)
    else:
        run_images(args)


if __name__ == "__main__":
    main()
