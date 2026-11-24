# -*- coding: utf-8 -*-
"""
MetalScan 工业零件表面缺陷检测系统 - 网页版 V3
================================================
启动:  D:/Apps/yolo/gpu_venv/Scripts/python.exe web_app.py
浏览器: http://localhost:8000

V3 新增模块:
  - 定时抓拍: 可设置取图间隔(2s~120s), 自动取流画面+检测+存档, 附抓拍画廊
  - 统计看板: 今日/累计检测量、NG 率、缺陷类别分布、来源分布、时段分布
  - 手动抓拍入档: 服务器端保存标注帧, 与定时抓拍同一画廊
  - NG 声音告警(前端), IoU 阈值参数, 设置服务端持久化(settings.json)

V2 改进:
  - 单一采集线程模型, 修复"摄像头关闭后打不开"问题
  - Apple 风格 UI, 侧边栏多页面导航
  - 批量检测 / 检测记录(持久化+CSV导出) / 系统设置
"""
import base64
import csv
import json
import os
import re
import shutil
import threading
import time
from collections import Counter, deque
from datetime import datetime

os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2
import numpy as np
import torch
from flask import Flask, Response, request, send_file
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "runs", "metalscan_yolo26", "weights", "best.pt")
RECORDS_CSV = os.path.join(BASE, "results", "records.csv")
IMAGES_DIR = os.path.join(BASE, "images")
CAPTURES_DIR = os.path.join(BASE, "results", "captures")
SETTINGS_PATH = os.path.join(BASE, "results", "settings.json")
HOST, PORT = "0.0.0.0", 8000

CLASS_NAMES = {0: "scratch", 1: "dent", 2: "stain"}
CLASS_COLORS_BGR = {0: (244, 133, 66), 1: (83, 168, 52), 2: (7, 193, 255)}  # BGR
DEFECT_CLASSES = {0, 1, 2}

app = Flask(__name__)

print("loading model:", MODEL_PATH)
model = YOLO(MODEL_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    model.to("cuda")
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    GPU_NAME = "CPU"
print("device:", DEVICE, GPU_NAME)


# ================= 设置持久化 =================
DEFAULT_SETTINGS = {"conf": 0.45, "iou": 0.45, "auto_enabled": False,
                    "auto_interval": 10, "save_raw": False}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                s.update({k: v for k, v in json.load(f).items() if k in DEFAULT_SETTINGS})
    except Exception as e:
        print("load settings failed:", e)
    return s


def save_settings_file(s):
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("save settings failed:", e)


SETTINGS = load_settings()


# ================= 台账 (内存 + CSV 持久化) =================
records = deque(maxlen=1000)
records_lock = threading.Lock()


def _load_records():
    if os.path.exists(RECORDS_CSV):
        try:
            with open(RECORDS_CSV, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    records.append(row)
        except Exception as e:
            print("load records failed:", e)


def add_record(source, verdict, dets, elapsed_ms):
    rec = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "verdict": verdict,
        "count": str(len(dets)),
        "detail": ", ".join(f"{d['name']} {d['conf']:.2f}" for d in dets[:4]) or "-",
        "ms": str(int(elapsed_ms)),
    }
    with records_lock:
        records.appendleft(rec)
    try:
        new_file = not os.path.exists(RECORDS_CSV)
        os.makedirs(os.path.dirname(RECORDS_CSV), exist_ok=True)
        with open(RECORDS_CSV, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if new_file:
                w.writeheader()
            w.writerow(rec)
    except Exception as e:
        print("save record failed:", e)


_load_records()


# ================= 推理与标注 =================
def infer(frame, conf, iou=0.45):
    return model.predict(frame, conf=conf, iou=iou, verbose=False)[0]


def annotate(frame, result):
    h, w = frame.shape[:2]
    dets = []
    for b in result.boxes:
        cid = int(b.cls[0])
        if cid not in DEFECT_CLASSES:
            continue
        conf = float(b.conf[0])
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        color = CLASS_COLORS_BGR[cid]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        dets.append({"name": CLASS_NAMES[cid], "conf": conf, "box": [x1, y1, x2, y2]})
    verdict = "NG" if dets else "OK"
    # 顶部判定条
    vc = (48, 48, 255) if verdict == "NG" else (89, 199, 52)      # BGR: FF3B30 / 34C759
    band_h = max(36, int(h * 0.09))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), vc, -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    fs = max(0.7, band_h / 62)
    cv2.putText(frame, verdict, (16, int(band_h * 0.74)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 3, cv2.LINE_AA)
    sub = f"defects: {len(dets)}   {datetime.now().strftime('%H:%M:%S')}"
    cv2.putText(frame, sub, (int(w * 0.24), int(band_h * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, dets, verdict


def detect_image(img_bgr, conf, iou=0.45):
    """单图检测 -> (标注图, 明细, 判定, 耗时ms)"""
    t0 = time.time()
    result = infer(img_bgr, conf, iou)
    img, dets, verdict = annotate(img_bgr, result)
    return img, dets, verdict, (time.time() - t0) * 1000


# ================= 摄像头管理 (单一采集线程) =================
class CameraManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.error = None
        self.conf = SETTINGS["conf"]
        self.iou = SETTINGS["iou"]
        self.annotated = None      # 最新标注帧 (共享给所有客户端)
        self.raw = None            # 最新原始帧 (抓拍原图用)
        self.dets = []
        self.verdict = "OK"
        self.fps = 0.0
        self.last_ms = 0

    def status(self):
        with self.lock:
            return {"on": self.running, "error": self.error,
                    "verdict": self.verdict, "fps": round(self.fps, 1),
                    "conf": self.conf, "iou": self.iou}

    def start(self):
        with self.lock:
            if self.running:
                return True, "already running"
            self.error = None
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.running = True
            self.thread.start()
            return True, "started"

    def stop(self):
        with self.lock:
            was = self.running
            self.running = False
        if was and self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            self.annotated = None
            self.raw = None
            self.dets, self.verdict, self.fps = [], "OK", 0.0
        return True, "stopped"

    def _open_camera(self, retries=3):
        for i in range(retries):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if cap.isOpened() and (ok or i < retries - 1):
                # 预热再读一帧确认
                for _ in range(5):
                    ok2, _f = cap.read()
                    if ok2:
                        return cap
            cap.release()
            time.sleep(0.6)
        return None

    def _loop(self):
        cap = self._open_camera()
        if cap is None:
            with self.lock:
                self.running = False
                self.error = "无法打开摄像头 (可能被其他程序占用)"
            return
        fail = 0
        t_last, n = time.time(), 0
        # 台账防抖: 候选判定需稳定 0.6s 才记录; 稳定后每 3s 心跳记录一次
        cand_verdict, cand_since = None, 0.0
        stable_verdict, last_rec_time = None, 0.0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                fail += 1
                if fail > 30:
                    # 尝试热重连
                    cap.release()
                    cap = self._open_camera()
                    if cap is None:
                        with self.lock:
                            self.error = "摄像头读取失败, 已停止"
                            self.running = False
                        return
                    fail = 0
                time.sleep(0.05)
                continue
            fail = 0
            now = time.time()
            try:
                conf, iou = self.conf, self.iou
                t0 = time.time()
                raw = frame.copy()
                img, dets, verdict = annotate(frame, infer(frame, conf, iou))
                infer_ms = (time.time() - t0) * 1000
                with self.lock:
                    self.annotated = img
                    self.raw = raw
                    self.dets, self.verdict = dets, verdict
                    self.last_ms = infer_ms
                # 台账防抖记录
                if verdict != cand_verdict:
                    cand_verdict, cand_since = verdict, now
                elif now - cand_since >= 0.6 and verdict != stable_verdict:
                    stable_verdict = verdict
                    add_record("camera", verdict, dets, infer_ms)
                    last_rec_time = now
                elif stable_verdict is not None and now - last_rec_time >= 3.0:
                    add_record("camera", stable_verdict, dets, infer_ms)
                    last_rec_time = now
            except Exception as e:
                print("camera infer error:", e)
            n += 1
            if now - t_last >= 1.0:
                self.fps = n / (now - t_last)
                t_last, n = now, 0
        cap.release()
        print("camera released")


cam = CameraManager()


# ================= 定时抓拍模块 (实时取图时间设置) =================
class CaptureManager:
    """按设定间隔自动从摄像头取图 -> 检测判定 -> 存档(标注图+可选原图) -> 台账"""

    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = bool(SETTINGS.get("auto_enabled", False))
        self.interval = float(SETTINGS.get("auto_interval", 10))
        self.save_raw = bool(SETTINGS.get("save_raw", False))
        self.next_ts = time.time() + self.interval
        self.count = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def configure(self, enabled=None, interval=None, save_raw=None):
        with self.lock:
            if interval is not None:
                self.interval = max(2.0, min(float(interval), 600.0))
            if save_raw is not None:
                self.save_raw = bool(save_raw)
            if enabled is not None:
                en = bool(enabled)
                if en != self.enabled:
                    self.enabled = en
                    self.next_ts = time.time() + (2.0 if en else 0)
        return self.status()

    def status(self):
        with self.lock:
            return {"enabled": self.enabled, "interval": self.interval,
                    "save_raw": self.save_raw, "count": self.count,
                    "next_in": max(0, int(self.next_ts - time.time())) if self.enabled else None,
                    "waiting_camera": self.enabled and not cam.running}

    def trigger(self):
        """手动立即抓拍一张, 返回 (ok, msg, meta)"""
        with self.lock:
            save_raw = self.save_raw
        with cam.lock:
            img = None if cam.annotated is None else cam.annotated.copy()
            raw = (None if cam.raw is None else cam.raw.copy()) if save_raw else None
            dets, verdict, ms = list(cam.dets), cam.verdict, cam.last_ms
        if img is None:
            return False, "摄像头未开启, 无法抓拍", None
        meta = self._save(img, raw, dets, verdict, ms)
        return True, "已抓拍", meta

    def _save(self, img, raw, dets, verdict, ms):
        ts = datetime.now()
        base = ts.strftime("cap_%Y%m%d_%H%M%S_%f")[:-3]
        ann_name = base + "_ann.jpg"
        ann_path = os.path.join(CAPTURES_DIR, ann_name)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            buf.tofile(ann_path)
        raw_name = None
        if raw is not None:
            raw_name = base + "_raw.jpg"
            ok2, buf2 = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok2:
                buf2.tofile(os.path.join(CAPTURES_DIR, raw_name))
        meta = {"ann": ann_name, "raw": raw_name,
                "time": ts.strftime("%H:%M:%S"), "date": ts.strftime("%Y-%m-%d"),
                "verdict": verdict, "count": len(dets),
                "detail": "；".join(f"{d['name']} {d['conf']:.2f}" for d in dets) or "未检出缺陷",
                "ms": int(ms)}
        try:
            with open(os.path.join(CAPTURES_DIR, base + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception as e:
            print("save capture meta failed:", e)
        add_record("capture", verdict, dets, ms)
        with self.lock:
            self.count += 1
        return meta

    def _loop(self):
        while True:
            time.sleep(0.4)
            with self.lock:
                if not self.enabled:
                    continue
                interval, due = self.interval, time.time() >= self.next_ts
            if not due:
                continue
            ok, _msg, _meta = self.trigger()
            with self.lock:
                self.next_ts = time.time() + interval


cap_mgr = CaptureManager()
os.makedirs(CAPTURES_DIR, exist_ok=True)


def _list_captures(limit=60):
    """扫描抓拍目录, 按 json sidecar 汇总, 最新在前"""
    out = []
    try:
        for fn in sorted(os.listdir(CAPTURES_DIR), reverse=True):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(CAPTURES_DIR, fn), encoding="utf-8") as f:
                    m = json.load(f)
                if os.path.exists(os.path.join(CAPTURES_DIR, m.get("ann", ""))):
                    out.append(m)
            except Exception:
                continue
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def placeholder_frame(text):
    img = np.zeros((480, 800, 3), np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, text, (230, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
    return img


def mjpeg_stream():
    while True:
        with cam.lock:
            img = None if cam.annotated is None else cam.annotated.copy()
        if img is None:
            st = cam.status()
            img = placeholder_frame("CAMERA OFF - click camera button to start")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)


# ================= 批量检测 =================
def batch_detect_images(items, conf, source_tag, iou=0.45):
    """items: [(name, bgr_img)]; 返回结果列表"""
    out = []
    for name, img in items:
        if max(img.shape[:2]) > 1600:
            s = 1600 / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        ann, dets, verdict, ms = detect_image(img, conf, iou)
        ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
        add_record(source_tag, verdict, dets, ms)
        out.append({
            "name": name, "verdict": verdict, "count": len(dets), "ms": int(ms),
            "detail": "；".join(f"{d['name']} {d['conf']:.2f}" for d in dets) or "未检出缺陷",
            "image": base64.b64encode(buf.tobytes()).decode(),
        })
    return out


# ================= 页面 =================
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MetalScan 工业零件表面缺陷检测系统</title>
<style>
  :root {
    --bg:#f5f5f7; --card:#ffffff; --line:rgba(0,0,0,.08); --txt:#1d1d1f; --muted:#86868b;
    --accent:#007aff; --ok:#34c759; --ng:#ff3b30; --warn:#ff9500;
    --shadow:0 4px 24px rgba(0,0,0,.06); --radius:18px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; }
  body { background:var(--bg); color:var(--txt);
         font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","PingFang SC","Microsoft YaHei",sans-serif;
         -webkit-font-smoothing:antialiased; }
  /* ---------- 布局: 左侧栏 + 主区 (对称) ---------- */
  .layout { display:flex; min-height:100vh; }
  aside { width:230px; flex-shrink:0; padding:24px 14px;
          background:rgba(255,255,255,.72); backdrop-filter:blur(20px);
          border-right:1px solid var(--line); position:sticky; top:0; height:100vh;
          display:flex; flex-direction:column; }
  .brand { display:flex; align-items:center; gap:10px; padding:0 10px 22px; }
  .brand .logo { width:36px; height:36px; border-radius:9px; flex-shrink:0;
                 background:linear-gradient(135deg,#007aff,#5e5ce6);
                 display:flex; align-items:center; justify-content:center;
                 color:#fff; font-weight:700; font-size:14px; letter-spacing:.5px; }
  .brand h1 { font-size:16px; font-weight:600; }
  .brand p { font-size:11px; color:var(--muted); }
  nav { display:flex; flex-direction:column; gap:2px; }
  nav a { display:flex; align-items:center; gap:10px; padding:9px 12px; border-radius:10px;
          color:var(--txt); text-decoration:none; font-size:14px; font-weight:500;
          transition:background .15s; }
  nav a:hover { background:rgba(0,0,0,.05); }
  nav a.active { background:var(--accent); color:#fff; }
  nav a .ic { width:20px; text-align:center; font-size:16px; }
  .sidebar-foot { margin-top:auto; padding:10px 12px; font-size:11px; color:var(--muted); line-height:1.7; }
  .cam-switch { display:flex; align-items:center; gap:8px; margin:14px 12px 0; padding:10px 12px;
                background:var(--card); border-radius:12px; box-shadow:var(--shadow); font-size:13px; }
  .switch { position:relative; width:42px; height:26px; flex-shrink:0; }
  .switch input { opacity:0; width:0; height:0; }
  .slider { position:absolute; inset:0; background:#d1d1d6; border-radius:26px; cursor:pointer; transition:.25s; }
  .slider:before { content:""; position:absolute; width:22px; height:22px; border-radius:50%;
                   background:#fff; left:2px; top:2px; transition:.25s; box-shadow:0 1px 3px rgba(0,0,0,.25); }
  .switch input:checked + .slider { background:var(--ok); }
  .switch input:checked + .slider:before { transform:translateX(16px); }
  main { flex:1; min-width:0; padding:28px 36px 48px; max-width:1180px; margin:0 auto; }
  .page { display:none; }  .page.active { display:block; }
  .page-head { margin-bottom:22px; }
  .page-head h2 { font-size:26px; font-weight:700; letter-spacing:-.3px; }
  .page-head p { color:var(--muted); font-size:13px; margin-top:4px; }
  .card { background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow);
          padding:22px; border:1px solid var(--line); }
  .grid-2 { display:grid; grid-template-columns:7fr 5fr; gap:20px; align-items:start; }
  @media (max-width:1000px){ .grid-2 { grid-template-columns:1fr; } .layout aside { width:200px; } }
  .card h3 { font-size:16px; font-weight:600; margin-bottom:14px; }
  /* 视频流 */
  .stream-wrap { position:relative; border-radius:14px; overflow:hidden; background:#111; }
  .stream-wrap img { width:100%; display:block; aspect-ratio:16/9.6; object-fit:cover; }
  .live-pill { position:absolute; top:12px; right:12px; display:flex; align-items:center; gap:6px;
               padding:5px 12px; border-radius:20px; font-size:12px; font-weight:600;
               background:rgba(0,0,0,.45); color:#fff; backdrop-filter:blur(8px); }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--muted); }
  .dot.on { background:var(--ng); animation:pulse 1.4s infinite; }
  @keyframes pulse { 50% { opacity:.4; } }
  /* 统计 */
  .stats { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
  .stat { background:var(--bg); border-radius:14px; padding:16px; }
  .stat .num { font-size:26px; font-weight:700; letter-spacing:-.5px; }
  .stat .lab { font-size:12px; color:var(--muted); margin-top:2px; }
  .stat.ok .num { color:var(--ok); } .stat.ng .num { color:var(--ng); }
  .meter { margin-top:14px; }
  .meter-bar { height:8px; border-radius:4px; background:var(--bg); overflow:hidden; display:flex; }
  .meter-bar .ok-b { background:var(--ok); } .meter-bar .ng-b { background:var(--ng); }
  /* 按钮 / 控件 */
  .btn { display:inline-flex; align-items:center; gap:6px; padding:9px 18px; border:none; border-radius:11px;
         background:var(--accent); color:#fff; cursor:pointer; font-size:14px; font-weight:500;
         font-family:inherit; transition:.15s; }
  .btn:hover { filter:brightness(1.08); }  .btn:active { transform:scale(.97); }
  .btn.sec { background:rgba(0,0,0,.06); color:var(--txt); }
  .btn:disabled { opacity:.45; cursor:not-allowed; }
  input[type=range] { accent-color:var(--accent); width:100%; }
  select { padding:8px 12px; border-radius:10px; border:1px solid var(--line); font-family:inherit;
           font-size:14px; background:var(--card); color:var(--txt); }
  /* 上传 */
  .drop { border:2px dashed #c7c7cc; border-radius:16px; padding:44px 20px; text-align:center;
          color:var(--muted); cursor:pointer; transition:.2s; background:var(--bg); }
  .drop:hover, .drop.over { border-color:var(--accent); background:rgba(0,122,255,.05); }
  .drop .big { font-size:34px; margin-bottom:8px; }
  .drop b { color:var(--txt); font-weight:600; }
  #resultImg { width:100%; border-radius:14px; display:none; margin-top:16px; box-shadow:var(--shadow); }
  .verdict-chip { display:inline-flex; align-items:center; padding:4px 16px; border-radius:20px;
                  font-weight:700; font-size:15px; color:#fff; }
  .verdict-chip.OK { background:var(--ok); } .verdict-chip.NG { background:var(--ng); }
  /* 批量 / 抓拍网格 */
  .batch-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:16px; }
  .batch-item { position:relative; border-radius:14px; overflow:hidden; background:var(--bg); border:1px solid var(--line); }
  .batch-item .cap-del { position:absolute; top:8px; right:8px; width:28px; height:28px; border-radius:50%;
      border:none; background:rgba(20,22,28,.55); color:#fff; font-size:14px; line-height:28px; text-align:center;
      cursor:pointer; opacity:0; transition:opacity .15s, background .15s; padding:0; }
  .batch-item:hover .cap-del { opacity:1; }
  .batch-item .cap-del:hover { background:var(--ng); }
  .batch-item img { width:100%; display:block; cursor:zoom-in; }
  .batch-item .bi-foot { padding:10px 12px; font-size:12px; display:flex; justify-content:space-between; align-items:center; }
  .batch-item .bi-foot .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:60%; color:var(--muted); }
  /* 表格 */
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:500; font-size:12px; }
  tr:hover td { background:rgba(0,122,255,.03); }
  .vOK { color:var(--ok); font-weight:600; } .vNG { color:var(--ng); font-weight:600; }
  .empty { color:var(--muted); font-size:13px; padding:28px; text-align:center; }
  .chips { display:flex; gap:12px; margin-bottom:18px; flex-wrap:wrap; }
  .chip { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 20px;
          box-shadow:var(--shadow); min-width:110px; }
  .chip .num { font-size:20px; font-weight:700; }
  .chip .lab { font-size:11px; color:var(--muted); }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  .toolbar { display:flex; gap:10px; align-items:center; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; }
  .toast { position:fixed; bottom:28px; left:50%; transform:translateX(-50%) translateY(80px);
           background:#1d1d1f; color:#fff; padding:11px 22px; border-radius:24px; font-size:13px;
           transition:.3s; z-index:99; opacity:0; box-shadow:0 8px 30px rgba(0,0,0,.3); }
  .toast.show { transform:translateX(-50%) translateY(0); opacity:1; }
  .kv { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line); font-size:13px; }
  .kv:last-child { border:none; }
  .kv span:first-child { color:var(--muted); }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid #fff5; border-top-color:#fff;
          border-radius:50%; animation:spin .7s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  /* 看板横条 */
  .bar-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; font-size:13px; }
  .bar-row .bl { width:110px; color:var(--muted); flex-shrink:0; }
  .bar-row .bar-track { flex:1; height:10px; background:var(--bg); border-radius:5px; overflow:hidden; }
  .bar-row .bar-fill { height:100%; border-radius:5px; background:var(--accent); transition:width .4s; }
  .bar-row .bv { width:56px; text-align:right; font-weight:600; flex-shrink:0; }
  .lightbox { position:fixed; inset:0; background:rgba(0,0,0,.72); display:none;
              align-items:center; justify-content:center; z-index:100; backdrop-filter:blur(6px); }
  .lightbox.show { display:flex; }
  .lightbox img { max-width:88vw; max-height:82vh; border-radius:14px; box-shadow:0 20px 60px rgba(0,0,0,.5); }
  .lightcap { position:fixed; bottom:26px; left:50%; transform:translateX(-50%); color:#fff;
              font-size:13px; background:rgba(0,0,0,.5); padding:8px 18px; border-radius:20px; }
  .cd-num { font-size:40px; font-weight:700; letter-spacing:-1px; }
  .cd-num.idle { color:var(--muted); }
</style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand">
      <div class="logo">MS</div>
      <div><h1>MetalScan</h1><p>工业缺陷检测 v3.0</p></div>
    </div>
    <nav id="nav">
      <a class="active" data-page="live" href="#"><span class="ic">◉</span>实时检测</a>
      <a data-page="capture" href="#"><span class="ic">⏱</span>定时抓拍</a>
      <a data-page="single" href="#"><span class="ic">◫</span>图片检测</a>
      <a data-page="batch" href="#"><span class="ic">▦</span>批量检测</a>
      <a data-page="records" href="#"><span class="ic">☰</span>检测记录</a>
      <a data-page="stats" href="#"><span class="ic">◫</span>统计看板</a>
      <a data-page="settings" href="#"><span class="ic">⚙</span>系统设置</a>
    </nav>
    <div class="cam-switch">
      <label class="switch"><input type="checkbox" id="camToggle"><span class="slider"></span></label>
      <div><div style="font-weight:600" id="camState">摄像头关闭</div><div style="font-size:11px;color:var(--muted)" id="camFps">— fps</div></div>
    </div>
    <div class="sidebar-foot">YOLO26n · NEU-DET + GC10-DET<br>四分类 · 本地 GPU 推理<br>© 2026 MetalScan</div>
  </aside>

  <main>
    <!-- ===== 实时检测 ===== -->
    <section class="page active" id="page-live">
      <div class="page-head"><h2>实时检测</h2><p>摄像头画面 · 服务端 GPU 实时推理 · 自动判定 OK / NG · 支持 NG 声音告警</p></div>
      <div class="grid-2">
        <div class="card">
          <div class="stream-wrap">
            <img id="stream" src="" alt="camera stream">
            <span class="live-pill"><span class="dot" id="liveDot"></span><span id="liveTxt">LIVE</span></span>
          </div>
          <div class="row" style="margin-top:14px">
            <span style="font-size:13px;color:var(--muted)">置信度 <b id="confLive">0.45</b></span>
            <input type="range" id="confLiveSlider" min="0.15" max="0.9" step="0.05" value="0.45" style="flex:1">
            <button class="btn sec" id="snapshotBtn">⬇ 下载当前帧</button>
            <button class="btn sec" id="archiveBtn">📷 抓拍入档</button>
          </div>
          <div class="row" style="margin-top:12px">
            <label class="switch" style="transform:scale(.85)"><input type="checkbox" id="ngAlarm"><span class="slider"></span></label>
            <span style="font-size:13px;color:var(--muted)">NG 声音告警</span>
            <span style="font-size:12px;color:var(--muted);margin-left:auto" id="lastBeepInfo"></span>
          </div>
        </div>
        <div class="card">
          <h3>当前画面状态</h3>
          <div class="stats">
            <div class="stat"><div class="num" id="stVerdict">—</div><div class="lab">当前判定</div></div>
            <div class="stat"><div class="num" id="stDefects" style="color:var(--txt)">0</div><div class="lab">缺陷数量</div></div>
            <div class="stat ok"><div class="num" id="stOk">0</div><div class="lab">本页 OK 次数</div></div>
            <div class="stat ng"><div class="num" id="stNg">0</div><div class="lab">本页 NG 次数</div></div>
          </div>
          <div class="meter">
            <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:6px">
              <span>OK 占比</span><span id="stRate">—</span></div>
            <div class="meter-bar"><div class="ok-b" id="okBar" style="width:50%"></div><div class="ng-b" id="ngBar" style="width:50%"></div></div>
          </div>
        </div>
      </div>
    </section>

    <!-- ===== 定时抓拍 ===== -->
    <section class="page" id="page-capture">
      <div class="page-head"><h2>定时抓拍</h2><p>按设定间隔自动取图并检测存档 · 也可手动立即抓拍 · 图片持久保存在 results/captures</p></div>
      <div class="grid-2">
        <div class="card">
          <h3>取图设置</h3>
          <div style="margin-bottom:16px">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
              <span>取图间隔</span><b id="capIntervalVal">10 秒</b></div>
            <select id="capInterval" style="width:100%">
              <option value="2">每 2 秒（高频巡检）</option>
              <option value="5">每 5 秒</option>
              <option value="10" selected>每 10 秒</option>
              <option value="30">每 30 秒</option>
              <option value="60">每 1 分钟</option>
              <option value="120">每 2 分钟</option>
            </select>
          </div>
          <div class="row" style="margin-bottom:16px">
            <label class="switch" style="transform:scale(.85)"><input type="checkbox" id="capRaw"><span class="slider"></span></label>
            <span style="font-size:13px">同时保存原始图（无标注帧）</span>
          </div>
          <div class="row">
            <button class="btn" id="capToggleBtn">▶ 开启定时抓拍</button>
            <button class="btn sec" id="capNowBtn">📷 立即抓拍一张</button>
          </div>
        </div>
        <div class="card">
          <h3>抓拍状态</h3>
          <div class="stats" style="grid-template-columns:repeat(2,1fr)">
            <div class="stat"><div class="num cd-num idle" id="capCountdown">—</div><div class="lab">距下次取图（秒）</div></div>
            <div class="stat"><div class="num" id="capCount" style="color:var(--txt)">0</div><div class="lab">本次运行已抓拍</div></div>
            <div class="stat ok"><div class="num" id="capOk">0</div><div class="lab">抓拍 OK</div></div>
            <div class="stat ng"><div class="num" id="capNg">0</div><div class="lab">抓拍 NG</div></div>
          </div>
          <div style="margin-top:12px;font-size:12px;color:var(--muted)" id="capStateTxt">定时抓拍未开启</div>
        </div>
      </div>
      <div class="card" style="margin-top:20px">
        <div class="toolbar">
          <h3 style="margin:0">抓拍画廊 <span style="font-size:12px;color:var(--muted);font-weight:400">（点击图片放大 · 悬停显示删除按钮 · 最新在前，最多显示 60 张）</span></h3>
          <div style="display:flex;gap:8px">
            <button class="btn sec" id="capRefresh">刷新</button>
            <button class="btn sec" id="capClear" style="color:var(--ng)">清空画廊</button>
          </div>
        </div>
        <div class="batch-grid" id="capGrid"><div class="empty" style="grid-column:1/-1">暂无抓拍图片</div></div>
      </div>
    </section>

    <!-- ===== 图片检测 ===== -->
    <section class="page" id="page-single">
      <div class="page-head"><h2>图片检测</h2><p>上传单张零件图片，获取标注结果与判定明细</p></div>
      <div class="grid-2">
        <div class="card">
          <div class="drop" id="drop">
            <div class="big">🖼️</div>
            <b>点击选择图片</b> 或将图片拖拽到这里<br>
            <span style="font-size:12px">支持 jpg / png，单张最大 10MB</span>
          </div>
          <input type="file" id="file" accept="image/*" hidden>
          <img id="resultImg" alt="result">
        </div>
        <div class="card">
          <h3>检测结果</h3>
          <div id="singleInfo" class="empty">尚未检测，左侧上传图片后此处显示结果</div>
        </div>
      </div>
    </section>

    <!-- ===== 批量检测 ===== -->
    <section class="page" id="page-batch">
      <div class="page-head"><h2>批量检测</h2><p>一次上传多张图片，或直接检测服务器 images 目录中的样例</p></div>
      <div class="card">
        <div class="toolbar">
          <div class="row">
            <button class="btn" id="batchUploadBtn">⬆ 上传多张图片</button>
            <select id="batchFolderSel">
              <option value="all">images 全部类别</option>
              <option value="scratch">仅 scratch（划痕）</option>
              <option value="dent">仅 dent（缺角）</option>
              <option value="stain">仅 stain（脏污）</option>
              <option value="normal_part">仅 normal_part（正常件）</option>
            </select>
            <button class="btn sec" id="batchFolderBtn">检测服务器样例（每类 6 张）</button>
          </div>
          <span style="font-size:12px;color:var(--muted)" id="batchHint"></span>
        </div>
        <div id="batchSummary" class="chips" style="display:none"></div>
        <div class="batch-grid" id="batchGrid"></div>
      </div>
      <input type="file" id="batchFile" accept="image/*" multiple hidden>
    </section>

    <!-- ===== 检测记录 ===== -->
    <section class="page" id="page-records">
      <div class="page-head"><h2>检测记录</h2><p>所有检测自动存档（CSV 持久化，重启不丢失）</p></div>
      <div class="chips">
        <div class="chip"><div class="num" id="rcTotal">0</div><div class="lab">总检测次数</div></div>
        <div class="chip"><div class="num" style="color:var(--ok)" id="rcOk">0</div><div class="lab">OK</div></div>
        <div class="chip"><div class="num" style="color:var(--ng)" id="rcNg">0</div><div class="lab">NG</div></div>
        <div class="chip"><div class="num" id="rcRate">—</div><div class="lab">NG 率</div></div>
      </div>
      <div class="card">
        <div class="toolbar">
          <h3 style="margin:0">最近记录</h3>
          <div class="row">
            <select id="rcFilter">
              <option value="">全部来源</option><option value="camera">摄像头</option>
              <option value="capture">定时抓拍</option>
              <option value="upload">单图上传</option><option value="batch">批量检测</option>
            </select>
            <button class="btn sec" onclick="location.href='/api/export'">⬇ 导出 CSV</button>
            <button class="btn sec" id="rcClear">清空记录</button>
          </div>
        </div>
        <table>
          <thead><tr><th>日期</th><th>时间</th><th>来源</th><th>判定</th><th>缺陷数</th><th>明细</th><th>耗时</th></tr></thead>
          <tbody id="rcBody"><tr><td colspan="7" class="empty">暂无记录</td></tr></tbody>
        </table>
      </div>
    </section>

    <!-- ===== 统计看板 ===== -->
    <section class="page" id="page-stats">
      <div class="page-head"><h2>统计看板</h2><p>基于全部检测记录自动汇总 · 每 5 秒刷新</p></div>
      <div class="chips">
        <div class="chip"><div class="num" id="sbToday">0</div><div class="lab">今日检测</div></div>
        <div class="chip"><div class="num" style="color:var(--ng)" id="sbTodayNg">0</div><div class="lab">今日 NG</div></div>
        <div class="chip"><div class="num" id="sbTotal">0</div><div class="lab">累计检测</div></div>
        <div class="chip"><div class="num" id="sbNgRate">—</div><div class="lab">累计 NG 率</div></div>
        <div class="chip"><div class="num" id="sbCaptures">0</div><div class="lab">抓拍图片</div></div>
      </div>
      <div class="grid-2">
        <div class="card">
          <h3>缺陷类别分布（累计检出缺陷次数）</h3>
          <div id="sbCat"></div>
        </div>
        <div class="card">
          <h3>检测来源分布</h3>
          <div id="sbSrc"></div>
        </div>
        <div class="card">
          <h3>今日时段分布（0–23 时检测次数）</h3>
          <div id="sbHour" style="display:flex;align-items:flex-end;gap:4px;height:150px"></div>
        </div>
        <div class="card">
          <h3>今日 OK / NG 占比</h3>
          <div style="display:flex;justify-content:center;padding:14px 0">
            <div style="width:170px;height:170px;border-radius:50%;position:relative" id="sbDonut"></div>
          </div>
          <div style="display:flex;justify-content:center;gap:18px;font-size:13px" id="sbDonutLegend"></div>
        </div>
      </div>
    </section>

    <!-- ===== 系统设置 ===== -->
    <section class="page" id="page-settings">
      <div class="page-head"><h2>系统设置</h2><p>默认参数、定时抓拍默认值与运行环境（保存至服务端 settings.json）</p></div>
      <div class="grid-2">
        <div class="card">
          <h3>检测参数</h3>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
              <span>默认置信度阈值</span><b id="confSettingVal">0.45</b></div>
            <input type="range" id="confSetting" min="0.15" max="0.9" step="0.05" value="0.45">
          </div>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
              <span>NMS IoU 阈值</span><b id="iouSettingVal">0.45</b></div>
            <input type="range" id="iouSetting" min="0.2" max="0.85" step="0.05" value="0.45">
            <div style="font-size:12px;color:var(--muted);margin-top:6px">IoU 越低重叠框过滤越强；缺陷密集时可调高以减少合并</div>
          </div>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
              <span>定时抓拍默认间隔</span><b id="capSettingVal">10 秒</b></div>
            <select id="capSetting" style="width:100%">
              <option value="2">每 2 秒</option><option value="5">每 5 秒</option>
              <option value="10" selected>每 10 秒</option><option value="30">每 30 秒</option>
              <option value="60">每 1 分钟</option><option value="120">每 2 分钟</option>
            </select>
          </div>
          <div class="row">
            <button class="btn" id="saveSetting">保存设置</button>
            <button class="btn sec" id="resetSetting">恢复默认</button>
          </div>
        </div>
        <div class="card">
          <h3>模型与环境</h3>
          <div id="sysInfo"></div>
        </div>
      </div>
    </section>
  </main>
</div>
<div class="toast" id="toast"></div>
<div class="lightbox" id="lightbox">
  <img id="lightboxImg" alt="preview">
  <div class="lightcap" id="lightboxCap"></div>
</div>
<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
function toast(msg){ const t=$('#toast'); t.textContent=msg; t.classList.add('show');
  clearTimeout(t._tm); t._tm=setTimeout(()=>t.classList.remove('show'),2200); }

/* ---------- 页面导航 ---------- */
function goPage(name){
  const a = document.querySelector(`#nav a[data-page="${name}"]`); if(!a) return false;
  $$('#nav a').forEach(x=>x.classList.remove('active')); a.classList.add('active');
  $$('.page').forEach(p=>p.classList.remove('active'));
  $('#page-'+a.dataset.page).classList.add('active');
  if(a.dataset.page==='records') loadRecords();
  if(a.dataset.page==='stats') loadStats();
  if(a.dataset.page==='capture') loadCaptures();
  return true;
}
$('#nav').addEventListener('click', e=>{
  const a = e.target.closest('a'); if(!a) return; e.preventDefault();
  goPage(a.dataset.page);
  history.replaceState(null,'','#p='+a.dataset.page);
});
window.addEventListener('DOMContentLoaded', ()=>{
  const m = location.hash.match(/^#p=(\w+)$/);
  if(m && goPage(m[1])) return;
});

/* ---------- 摄像头开关 ---------- */
const streamImg = $('#stream');
let streamOn = false;
function updateStream(){
  streamOn = $('#camToggle').checked;
  if(streamOn) streamImg.src = '/video_feed?t=' + Date.now();
  else streamImg.src = '';
  $('#camState').textContent = streamOn ? '摄像头运行中' : '摄像头关闭';
  $('#liveDot').classList.toggle('on', streamOn);
  $('#liveTxt').textContent = streamOn ? 'LIVE' : 'OFF';
}
$('#camToggle').addEventListener('change', async ()=>{
  const on = $('#camToggle').checked;
  const r = await fetch(on?'/api/camera/on':'/api/camera/off', {method:'POST'});
  const j = await r.json();
  if(!j.ok){ toast('摄像头启动失败: '+(j.msg||'')); $('#camToggle').checked=false; }
  updateStream();
});

/* ---------- NG 声音告警 ---------- */
let audioCtx = null;
function beepNG(){
  try{
    audioCtx = audioCtx || new (window.AudioContext||window.webkitAudioContext)();
    const t = audioCtx.currentTime;
    [0, .28].forEach(off=>{
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type='square'; o.frequency.value=880;
      g.gain.setValueAtTime(.001,t+off); g.gain.exponentialRampToValueAtTime(.22,t+off+.02);
      g.gain.exponentialRampToValueAtTime(.001,t+off+.18);
      o.connect(g); g.connect(audioCtx.destination);
      o.start(t+off); o.stop(t+off+.2);
    });
    $('#lastBeepInfo').textContent = '⚠ ' + new Date().toLocaleTimeString() + ' NG 告警';
  }catch(e){}
}

async function pollCamera(){
  try{
    const j = await (await fetch('/api/camera/status')).json();
    $('#camFps').textContent = j.on ? (j.fps+' fps') : '— fps';
    if(j.on){
      const prev = $('#stVerdict').textContent;
      $('#stVerdict').textContent = j.verdict; $('#stVerdict').style.color = j.verdict==='NG'?'var(--ng)':'var(--ok)';
      if($('#ngAlarm').checked && prev!=='NG' && j.verdict==='NG') beepNG();
      const r = await (await fetch('/api/camera/live')).json();
      $('#stDefects').textContent = r.count;
      r.items.forEach(d=>{ if(!d._seen){ d._seen=1; if(d.verdict==='OK')$('#stOk').textContent++; else $('#stNg').textContent++;
        const ok=+$('#stOk').textContent, ng=+$('#stNg').textContent, t=ok+ng;
        $('#stRate').textContent = Math.round(ok/t*100)+'%';
        $('#okBar').style.width=(ok/t*100)+'%'; $('#ngBar').style.width=(ng/t*100)+'%'; }});
      window._liveSeen = r.items; }
  }catch(e){}
}
setInterval(pollCamera, 1500);
$('#snapshotBtn').onclick = ()=>{ if(!streamOn) return toast('请先开启摄像头');
  const a=document.createElement('a'); a.href=streamImg.src; a.download='metalscan_'+Date.now()+'.jpg'; a.click(); };
$('#archiveBtn').onclick = async ()=>{
  if(!streamOn) return toast('请先开启摄像头');
  const j = await (await fetch('/api/capture/now',{method:'POST'})).json();
  toast(j.ok ? '已抓拍入档' : (j.msg||'抓拍失败'));
  if(j.ok) loadCaptures();
};

/* ---------- 灯箱 ---------- */
function openLightbox(src, cap){
  $('#lightboxImg').src = src; $('#lightboxCap').textContent = cap||'';
  $('#lightbox').classList.add('show');
}
$('#lightbox').onclick = ()=>$('#lightbox').classList.remove('show');

/* ---------- 定时抓拍 ---------- */
$('#capInterval').onchange = async e=>{
  await postSettings({auto_interval:+e.target.value});
  $('#capIntervalVal').textContent = '每 ' + e.target.value + ' 秒';
  pollCapture();
};
$('#capRaw').onchange = async e=>{ await postSettings({save_raw:e.target.checked}); pollCapture(); };
$('#capToggleBtn').onclick = async ()=>{
  const j = await (await fetch('/api/capture/config',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:!capState.enabled})})).json();
  capState = j; renderCapState();
};
$('#capNowBtn').onclick = async ()=>{
  const j = await (await fetch('/api/capture/now',{method:'POST'})).json();
  toast(j.ok ? '已抓拍入档' : (j.msg||'抓拍失败'));
  if(j.ok){ pollCapture(); loadCaptures(); }
};
$('#capRefresh').onclick = loadCaptures;
$('#capClear').onclick = async () => {
  if(!confirm('确定清空全部抓拍图片？该操作不可恢复（标注图、原图与元数据一并删除）')) return;
  try{
    await fetch('/api/captures/clear', {method:'POST'});
    loadCaptures(); pollCapture();
  }catch(e){ alert('清空失败：'+e); }
};
async function delCapture(ann, ev){
  ev.stopPropagation();
  if(!confirm('确定删除这张抓拍图片？')) return;
  try{
    const r = await fetch('/api/captures/delete', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({ann})});
    const j = await r.json();
    if(j.ok){ loadCaptures(); pollCapture(); }
    else alert('删除失败：'+(j.msg||r.status));
  }catch(e){ alert('删除失败：'+e); }
}

let capState = {enabled:false, interval:10, count:0, next_in:null, waiting_camera:false};
function renderCapState(){
  $('#capToggleBtn').textContent = capState.enabled ? '⏸ 停止定时抓拍' : '▶ 开启定时抓拍';
  $('#capCount').textContent = capState.count;
  const cd = $('#capCountdown');
  if(!capState.enabled){ cd.textContent='—'; cd.classList.add('idle');
    $('#capStateTxt').textContent='定时抓拍未开启'; return; }
  cd.classList.remove('idle');
  if(capState.waiting_camera){ cd.textContent='…'; $('#capStateTxt').textContent='等待摄像头开启（请打开左下角摄像头开关）'; }
  else { $('#capStateTxt').textContent='定时抓拍运行中 · 间隔 '+capState.interval+' 秒 · 存档于 results/captures'; }
}
async function pollCapture(){
  try{
    capState = await (await fetch('/api/capture/status')).json();
    renderCapState();
    if(capState.enabled && capState.next_in!=null){
      $('#capCountdown').textContent = capState.next_in;
      const cd=$('#capCountdown'); cd.classList.toggle('idle', capState.waiting_camera);
    }
    const caps = await (await fetch('/api/captures')).json();
    const ok = caps.filter(c=>c.verdict==='OK').length;
    $('#capOk').textContent = ok; $('#capNg').textContent = caps.length-ok;
  }catch(e){}
}
setInterval(pollCapture, 1000);
async function loadCaptures(){
  const caps = await (await fetch('/api/captures')).json();
  const g = $('#capGrid');
  g.innerHTML = caps.length ? caps.map(c=>`<div class="batch-item">
    <img src="/api/captures/file/${c.ann}" onclick="openLightbox('/api/captures/file/${c.ann}','${c.date} ${c.time} · ${c.verdict} · ${c.detail}')">
    <button class="cap-del" title="删除此抓拍" onclick="delCapture('${c.ann}',event)">✕</button>
    <div class="bi-foot"><span class="nm">${c.time}</span>
      <span class="v${c.verdict}">${c.verdict}</span>
      <span style="color:var(--muted)">${c.count>0?c.count+' 缺陷':'OK'}</span></div></div>`).join('')
    : '<div class="empty" style="grid-column:1/-1">暂无抓拍图片</div>';
}

/* ---------- 图片检测 ---------- */
const drop=$('#drop'), fileInput=$('#file');
drop.onclick=()=>fileInput.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over');};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');if(e.dataTransfer.files[0])uploadOne(e.dataTransfer.files[0]);};
fileInput.onchange=()=>{if(fileInput.files[0])uploadOne(fileInput.files[0]);};
async function uploadOne(f){
  $('#singleInfo').innerHTML='<span class="spin"></span> 检测中…'; $('#singleInfo').classList.remove('empty');
  const fd=new FormData(); fd.append('image',f);
  const j=await (await fetch('/api/detect',{method:'POST',body:fd})).json();
  const img=$('#resultImg'); img.src='data:image/jpeg;base64,'+j.image; img.style.display='block';
  $('#singleInfo').innerHTML=`<span class="verdict-chip ${j.verdict}">${j.verdict}</span>
    <div style="margin:14px 0 4px;font-size:26px;font-weight:700">${j.verdict==='NG'?'检出 '+j.count+' 处缺陷':'未检出缺陷'}</div>
    <div style="font-size:13px;color:var(--muted)">推理耗时 ${j.ms} ms · 置信度 ${j.conf} · IoU ${j.iou}</div>
    ${j.dets.map(d=>`<div class="kv"><span>${d.name}</span><span>置信度 ${d.conf.toFixed(2)}</span></div>`).join('')||''}`;
  loadRecords();
}

/* ---------- 批量检测 ---------- */
$('#batchUploadBtn').onclick=()=>$('#batchFile').click();
$('#batchFile').onchange=()=>{ if($('#batchFile').files.length) runBatchUpload([...$('#batchFile').files].slice(0,12)); };
$('#batchFolderBtn').onclick=async ()=>{
  $('#batchHint').innerHTML='<span class="spin"></span> 检测中…';
  const sel=$('#batchFolderSel').value;
  const j=await (await fetch('/api/batch_folder',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({category:sel,per_class:6})})).json();
  renderBatch(j.results, j.summary); $('#batchHint').textContent=`共 ${j.results.length} 张样例`;
};
async function runBatchUpload(files){
  $('#batchHint').innerHTML='<span class="spin"></span> 检测中…';
  const fd=new FormData(); files.forEach(f=>fd.append('images',f));
  const j=await (await fetch('/api/batch_upload',{method:'POST',body:fd})).json();
  renderBatch(j.results, j.summary); $('#batchHint').textContent=`共 ${j.results.length} 张`;
  loadRecords();
}
function renderBatch(results, summary){
  const g=$('#batchGrid');
  g.innerHTML=results.map(r=>`<div class="batch-item">
    <img src="data:image/jpeg;base64,${r.image}" onclick="openLightbox(this.src,'${r.name}')">
    <div class="bi-foot"><span class="nm" title="${r.name}">${r.name}</span>
      <span class="v${r.verdict}">${r.verdict}</span>
      <span style="color:var(--muted)">${r.count>0?r.count+' 缺陷':'OK'} · ${r.ms}ms</span></div></div>`).join('');
  const s=$('#batchSummary'); s.style.display='flex';
  s.innerHTML=`<div class="chip"><div class="num">${summary.total}</div><div class="lab">总数</div></div>
    <div class="chip"><div class="num" style="color:var(--ok)">${summary.ok}</div><div class="lab">OK</div></div>
    <div class="chip"><div class="num" style="color:var(--ng)">${summary.ng}</div><div class="lab">NG</div></div>
    <div class="chip"><div class="num">${summary.ngRate}</div><div class="lab">NG 率</div></div>`;
}

/* ---------- 检测记录 ---------- */
const SRC_LABEL = {camera:'📷 摄像头',capture:'⏱ 抓拍',upload:'🖼️ 单图',batch:'▦ 批量'};
async function loadRecords(){
  const rows=await (await fetch('/api/records')).json();
  const flt=$('#rcFilter').value;
  const shown=flt?rows.filter(r=>r.source===flt):rows;
  const ok=rows.filter(r=>r.verdict==='OK').length, ng=rows.length-ok;
  $('#rcTotal').textContent=rows.length; $('#rcOk').textContent=ok; $('#rcNg').textContent=ng;
  $('#rcRate').textContent=rows.length?Math.round(ng/rows.length*100)+'%':'—';
  $('#rcBody').innerHTML=shown.length?shown.slice(0,80).map(r=>`<tr>
    <td>${r.date}</td><td>${r.time}</td><td>${SRC_LABEL[r.source]||r.source}</td>
    <td class="v${r.verdict}">${r.verdict}</td><td>${r.count}</td><td>${r.detail}</td><td>${r.ms}ms</td></tr>`).join('')
    :'<tr><td colspan="7" class="empty">暂无记录</td></tr>';
}
$('#rcFilter').onchange=loadRecords;
$('#rcClear').onclick=async()=>{ if(!confirm('确认清空所有检测记录？'))return;
  await fetch('/api/records/clear',{method:'POST'}); toast('已清空'); loadRecords(); };

/* ---------- 统计看板 ---------- */
function barRows(container, data, colors){
  const max = Math.max(...data.map(d=>d.v), 1);
  container.innerHTML = data.map(d=>`<div class="bar-row">
    <span class="bl">${d.k}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${Math.round(d.v/max*100)}%;${d.c?`background:${d.c}`:''}"></div></div>
    <span class="bv">${d.v}</span></div>`).join('');
}
async function loadStats(){
  const s = await (await fetch('/api/stats')).json();
  $('#sbToday').textContent = s.today_total; $('#sbTodayNg').textContent = s.today_ng;
  $('#sbTotal').textContent = s.total; $('#sbNgRate').textContent = s.ng_rate;
  $('#sbCaptures').textContent = s.captures;
  barRows($('#sbCat'), [
    {k:'scratch 划痕', v:s.cat.scratch, c:'#f48542'},
    {k:'dent 缺角', v:s.cat.dent, c:'#53a834'},
    {k:'stain 脏污', v:s.cat.stain, c:'#07c1ff'}]);
  barRows($('#sbSrc'), [
    {k:'📷 摄像头', v:s.src.camera||0}, {k:'⏱ 定时抓拍', v:s.src.capture||0},
    {k:'🖼️ 单图', v:s.src.upload||0}, {k:'▦ 批量', v:s.src.batch||0}]);
  const hv = Array.from({length:24},(_,i)=>s.hours[i]||0);
  const hmax = Math.max(...hv, 1);
  $('#sbHour').innerHTML = hv.map((v,i)=>`<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:3px">
    <div style="font-size:10px;color:var(--muted)">${v||''}</div>
    <div style="width:100%;height:${v?Math.max(6,Math.round(v/hmax*110)):4}px;border-radius:4px 4px 0 0;background:${v?'var(--accent)':'var(--bg)'}"></div>
    <div style="font-size:9px;color:var(--muted)">${i}</div></div>`).join('');
  const t = s.today_total || 0, okn = t - s.today_ng;
  const okP = t ? Math.round(okn/t*100) : 0;
  $('#sbDonut').style.background = t ? `conic-gradient(var(--ok) 0 ${okP}%, var(--ng) ${okP}% 100%)` : 'var(--bg)';
  $('#sbDonut').style.boxShadow = 'inset 0 0 0 26px var(--card)';
  $('#sbDonutLegend').innerHTML = `<span style="color:var(--ok)">● OK ${okn}</span><span style="color:var(--ng)">● NG ${s.today_ng}</span>`;
}
setInterval(()=>{ if($('#page-stats').classList.contains('active')) loadStats(); }, 5000);

/* ---------- 设置 ---------- */
async function postSettings(patch){
  const j = await (await fetch('/api/settings',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)})).json();
  return j;
}
$('#confLiveSlider').oninput=e=>{ $('#confLive').textContent=e.target.value;
  postSettings({conf:parseFloat(e.target.value)}); };
$('#confSetting').oninput=e=>$('#confSettingVal').textContent=e.target.value;
$('#iouSetting').oninput=e=>$('#iouSettingVal').textContent=e.target.value;
$('#capSetting').onchange=e=>$('#capSettingVal').textContent='每 '+e.target.value+' 秒';
$('#saveSetting').onclick=async()=>{
  const conf=parseFloat($('#confSetting').value), iou=parseFloat($('#iouSetting').value);
  await postSettings({conf, iou, auto_interval:+$('#capSetting').value});
  $('#confLiveSlider').value=conf; $('#confLive').textContent=conf;
  $('#capIntervalVal').textContent='每 '+$('#capSetting').value+' 秒';
  localStorage.setItem('msConf',conf); toast('设置已保存（服务端持久化）');
  loadSysInfo();
};
$('#resetSetting').onclick=async()=>{
  await postSettings({conf:0.45, iou:0.45, auto_interval:10});
  $('#confSetting').value=0.45; $('#confSettingVal').textContent='0.45';
  $('#iouSetting').value=0.45; $('#iouSettingVal').textContent='0.45';
  $('#capSetting').value=10; $('#capSettingVal').textContent='每 10 秒';
  $('#confLiveSlider').value=0.45; $('#confLive').textContent='0.45';
  localStorage.removeItem('msConf'); toast('已恢复默认');
};
async function loadSysInfo(){
  const j = await (await fetch('/api/sysinfo')).json();
  $('#sysInfo').innerHTML=[['模型',j.model],['算法','YOLO26n · 100 epochs'],['mAP@0.5',j.map50],
    ['推理设备',j.device],['检测类别','scratch / dent / stain / normal_part'],['数据集',j.dataset],
    ['置信度 / IoU',j.conf+' / '+j.iou],['抓拍存档','results/captures'],
    ['访问地址','http://localhost:8000']].map(x=>`<div class="kv"><span>${x[0]}</span><span>${x[1]}</span></div>`).join('');
}
(async()=>{
  const s = await (await fetch('/api/settings')).json();
  $('#confSetting').value=s.conf; $('#confSettingVal').textContent=s.conf;
  $('#iouSetting').value=s.iou; $('#iouSettingVal').textContent=s.iou;
  $('#capSetting').value=s.auto_interval; $('#capSettingVal').textContent='每 '+s.auto_interval+' 秒';
  $('#capInterval').value=s.auto_interval; $('#capIntervalVal').textContent='每 '+s.auto_interval+' 秒';
  $('#capRaw').checked=s.save_raw;
  $('#confLiveSlider').value=s.conf; $('#confLive').textContent=s.conf;
  loadSysInfo();
})();
loadRecords();
pollCapture();
loadCaptures();
</script>
</body>
</html>"""


# ================= 路由 =================
@app.route("/")
def index():
    return PAGE_HTML


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/settings")
def api_settings():
    with cam.lock:
        s = dict(SETTINGS)
        s["conf"], s["iou"] = cam.conf, cam.iou
    with cap_mgr.lock:
        s["auto_enabled"], s["auto_interval"], s["save_raw"] = cap_mgr.enabled, int(cap_mgr.interval), cap_mgr.save_raw
    return s


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    data = request.get_json(force=True) or {}
    if "conf" in data:
        cam.conf = max(0.05, min(float(data["conf"]), 0.95))
        SETTINGS["conf"] = cam.conf
    if "iou" in data:
        cam.iou = max(0.1, min(float(data["iou"]), 0.9))
        SETTINGS["iou"] = cam.iou
    if "auto_interval" in data:
        cap_mgr.configure(interval=float(data["auto_interval"]))
        SETTINGS["auto_interval"] = int(cap_mgr.interval)
    if "save_raw" in data:
        cap_mgr.configure(save_raw=bool(data["save_raw"]))
        SETTINGS["save_raw"] = cap_mgr.save_raw
    if "auto_enabled" in data:
        cap_mgr.configure(enabled=bool(data["auto_enabled"]))
        SETTINGS["auto_enabled"] = cap_mgr.enabled
    save_settings_file(SETTINGS)
    return {"ok": True, "settings": SETTINGS}


@app.route("/api/capture/config", methods=["POST"])
def api_capture_config():
    data = request.get_json(force=True) or {}
    st = cap_mgr.configure(enabled=data.get("enabled"), interval=data.get("interval"))
    SETTINGS["auto_enabled"] = st["enabled"]
    SETTINGS["auto_interval"] = int(st["interval"])
    save_settings_file(SETTINGS)
    return st


@app.route("/api/capture/status")
def api_capture_status():
    return cap_mgr.status()


@app.route("/api/capture/now", methods=["POST"])
def api_capture_now():
    ok, msg, meta = cap_mgr.trigger()
    return {"ok": ok, "msg": msg, "meta": meta}


@app.route("/api/captures")
def api_captures():
    return _list_captures()


@app.route("/api/captures/file/<name>")
def api_capture_file(name):
    name = os.path.basename(name)
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "invalid file"}, 400
    p = os.path.join(CAPTURES_DIR, name)
    if not os.path.exists(p):
        return {"error": "not found"}, 404
    return send_file(p, mimetype="image/jpeg")


@app.route("/api/captures/delete", methods=["POST"])
def api_capture_delete():
    """删除一张抓拍: 连带清理原始图与 sidecar json"""
    data = request.get_json(force=True, silent=True) or {}
    ann = os.path.basename(str(data.get("ann", "")))
    if not ann.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"ok": False, "msg": "invalid file"}, 400
    base = ann[:-len("_ann.jpg")] if ann.lower().endswith("_ann.jpg") else os.path.splitext(ann)[0]
    removed = []
    for fn in (ann, base + "_raw.jpg", base + ".json"):
        p = os.path.join(CAPTURES_DIR, fn)
        if os.path.exists(p):
            try:
                os.remove(p)
                removed.append(fn)
            except Exception as e:
                print("delete capture failed:", e)
    if not removed:
        return {"ok": False, "msg": "文件不存在或已被删除"}, 404
    with cap_mgr.lock:
        if cap_mgr.count > 0:
            cap_mgr.count -= 1
    return {"ok": True, "msg": "已删除", "removed": len(removed)}


@app.route("/api/captures/clear", methods=["POST"])
def api_captures_clear():
    shutil.rmtree(CAPTURES_DIR, ignore_errors=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    with cap_mgr.lock:
        cap_mgr.count = 0
    return {"ok": True}


@app.route("/api/camera/on", methods=["POST"])
def api_cam_on():
    ok, msg = cam.start()
    return {"ok": ok, "msg": msg}


@app.route("/api/camera/off", methods=["POST"])
def api_cam_off():
    ok, msg = cam.stop()
    return {"ok": ok, "msg": msg}


@app.route("/api/camera/status")
def api_cam_status():
    return cam.status()


@app.route("/api/camera/live")
def api_cam_live():
    with cam.lock:
        return {"count": len(cam.dets),
                "items": [{"name": d["name"], "conf": d["conf"], "verdict": cam.verdict}
                          for d in cam.dets]}


@app.route("/api/camera/conf", methods=["POST"])
def api_cam_conf():
    try:
        cam.conf = float(request.get_json(force=True).get("conf", 0.45))
        SETTINGS["conf"] = cam.conf
        save_settings_file(SETTINGS)
        return {"ok": True, "conf": cam.conf}
    except Exception:
        return {"ok": False}, 400


@app.route("/api/detect", methods=["POST"])
def api_detect():
    f = request.files.get("image")
    if f is None:
        return {"error": "no image"}, 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image"}, 400
    if max(img.shape[:2]) > 1600:
        s = 1600 / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    ann, dets, verdict, ms = detect_image(img, cam.conf, cam.iou)
    add_record("upload", verdict, dets, ms)
    ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"verdict": verdict, "count": len(dets), "ms": int(ms),
            "conf": cam.conf, "iou": cam.iou,
            "dets": [{"name": d["name"], "conf": d["conf"]} for d in dets],
            "image": base64.b64encode(buf.tobytes()).decode()}


@app.route("/api/batch_upload", methods=["POST"])
def api_batch_upload():
    files = request.files.getlist("images")[:12]
    if not files:
        return {"error": "no images"}, 400
    items = []
    for f in files:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            items.append((f.filename, img))
    results = batch_detect_images(items, cam.conf, "batch", cam.iou)
    return {"results": results, "summary": _summary(results)}


@app.route("/api/batch_folder", methods=["POST"])
def api_batch_folder():
    data = request.get_json(force=True) or {}
    category = data.get("category", "all")
    per_class = min(int(data.get("per_class", 6)), 12)
    cats = [category] if category != "all" else ["scratch", "dent", "stain", "normal_part"]
    items = []
    for c in cats:
        cdir = os.path.join(IMAGES_DIR, c)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir))[:per_class]:
            p = os.path.join(cdir, fn)
            data_buf = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(data_buf, cv2.IMREAD_COLOR)
            if img is not None:
                items.append((f"{c}/{fn}", img))
    results = batch_detect_images(items, cam.conf, "batch", cam.iou)
    return {"results": results, "summary": _summary(results)}


def _summary(results):
    ok = sum(1 for r in results if r["verdict"] == "OK")
    ng = len(results) - ok
    return {"total": len(results), "ok": ok, "ng": ng,
            "ngRate": f"{round(ng/len(results)*100)}%" if results else "—"}


@app.route("/api/stats")
def api_stats():
    with records_lock:
        rows = list(records)
    today = datetime.now().strftime("%Y-%m-%d")
    today_rows = [r for r in rows if r["date"] == today]
    total, today_total = len(rows), len(today_rows)
    ng = sum(1 for r in rows if r["verdict"] == "NG")
    today_ng = sum(1 for r in today_rows if r["verdict"] == "NG")
    cat = {"scratch": 0, "dent": 0, "stain": 0}
    for r in rows:
        for m in re.findall(r"(scratch|dent|stain)\s+[\d.]+", r.get("detail", "") or ""):
            cat[m] += 1
    src = dict(Counter(r["source"] for r in rows))
    hours = dict(Counter(int(r["time"][:2]) for r in today_rows))
    captures = len(_list_captures(500))
    return {"today_total": today_total, "today_ng": today_ng, "total": total, "ng": ng,
            "ng_rate": f"{round(ng/total*100)}%" if total else "—",
            "cat": cat, "src": src, "hours": hours, "captures": captures}


@app.route("/api/records")
def api_records():
    with records_lock:
        return list(records)


@app.route("/api/records/clear", methods=["POST"])
def api_records_clear():
    with records_lock:
        records.clear()
    if os.path.exists(RECORDS_CSV):
        os.remove(RECORDS_CSV)
    return {"ok": True}


@app.route("/api/export")
def api_export():
    if not os.path.exists(RECORDS_CSV):
        return "no records", 404
    return send_file(RECORDS_CSV, as_attachment=True,
                     download_name=f"metalscan_records_{datetime.now():%Y%m%d_%H%M%S}.csv")


@app.route("/api/sysinfo")
def api_sysinfo():
    with cam.lock:
        conf, iou = cam.conf, cam.iou
    return {"model": os.path.basename(MODEL_PATH), "device": f"{DEVICE.upper()} ({GPU_NAME})",
            "map50": "0.874", "dataset": "NEU-DET + GC10-DET · 415 张",
            "conf": conf, "iou": iou}


if __name__ == "__main__":
    print(f"MetalScan web server V3: http://localhost:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)
