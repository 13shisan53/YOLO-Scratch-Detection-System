# -*- coding: utf-8 -*-
"""
MetalScan 工业零件表面缺陷检测系统 - 网页版 V2
================================================
启动:  D:/Apps/yolo/gpu_venv/Scripts/python.exe web_app.py
浏览器: http://localhost:8000

V2 改进:
  - 单一采集线程模型, 修复"摄像头关闭后打不开"问题
  - Apple 风格 UI, 侧边栏多页面导航
  - 新增: 批量检测 / 检测记录(持久化+CSV导出) / 系统设置 / 统计看板
"""
import base64
import csv
import os
import threading
import time
from datetime import datetime
from collections import deque

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


# ================= 台账 (内存 + CSV 持久化) =================
records = deque(maxlen=300)
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
def infer(frame, conf):
    return model.predict(frame, conf=conf, verbose=False)[0]


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


def detect_image(img_bgr, conf):
    """单图检测 -> (标注图, 明细, 判定, 耗时ms)"""
    t0 = time.time()
    result = infer(img_bgr, conf)
    img, dets, verdict = annotate(img_bgr, result)
    return img, dets, verdict, (time.time() - t0) * 1000


# ================= 摄像头管理 (单一采集线程) =================
class CameraManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.error = None
        self.conf = 0.45
        self.annotated = None      # 最新标注帧 (共享给所有客户端)
        self.dets = []
        self.verdict = "OK"
        self.fps = 0.0

    def status(self):
        with self.lock:
            return {"on": self.running, "error": self.error,
                    "verdict": self.verdict, "fps": round(self.fps, 1),
                    "conf": self.conf}

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
                conf = self.conf
                t0 = time.time()
                img, dets, verdict = annotate(frame, infer(frame, conf))
                infer_ms = (time.time() - t0) * 1000
                with self.lock:
                    self.annotated = img
                    self.dets, self.verdict = dets, verdict
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
def batch_detect_images(items, conf, source_tag):
    """items: [(name, bgr_img)]; 返回结果列表"""
    out = []
    for name, img in items:
        if max(img.shape[:2]) > 1600:
            s = 1600 / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        ann, dets, verdict, ms = detect_image(img, conf)
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
  nav a { display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:10px;
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
  /* 批量网格 */
  .batch-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:16px; }
  .batch-item { border-radius:14px; overflow:hidden; background:var(--bg); border:1px solid var(--line); }
  .batch-item img { width:100%; display:block; }
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
</style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand">
      <div class="logo">MS</div>
      <div><h1>MetalScan</h1><p>工业缺陷检测 v2.0</p></div>
    </div>
    <nav id="nav">
      <a class="active" data-page="live" href="#"><span class="ic">◉</span>实时检测</a>
      <a data-page="single" href="#"><span class="ic">◫</span>图片检测</a>
      <a data-page="batch" href="#"><span class="ic">▦</span>批量检测</a>
      <a data-page="records" href="#"><span class="ic">☰</span>检测记录</a>
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
      <div class="page-head"><h2>实时检测</h2><p>摄像头画面 · 服务端 GPU 实时推理 · 自动判定 OK / NG</p></div>
      <div class="grid-2">
        <div class="card">
          <div class="stream-wrap">
            <img id="stream" src="" alt="camera stream">
            <span class="live-pill"><span class="dot" id="liveDot"></span><span id="liveTxt">LIVE</span></span>
          </div>
          <div class="row" style="margin-top:14px">
            <span style="font-size:13px;color:var(--muted)">置信度 <b id="confLive">0.45</b></span>
            <input type="range" id="confLiveSlider" min="0.15" max="0.9" step="0.05" value="0.45" style="flex:1">
            <button class="btn sec" id="snapshotBtn">⬇ 保存当前帧</button>
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
            <select id="rcFilter"><option value="">全部来源</option><option value="camera">摄像头</option>
              <option value="upload">单图上传</option><option value="batch">批量检测</option></select>
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

    <!-- ===== 系统设置 ===== -->
    <section class="page" id="page-settings">
      <div class="page-head"><h2>系统设置</h2><p>默认参数、模型信息与运行环境</p></div>
      <div class="grid-2">
        <div class="card">
          <h3>默认参数</h3>
          <div style="margin-bottom:18px">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:6px">
              <span>默认置信度阈值</span><b id="confSettingVal">0.45</b></div>
            <input type="range" id="confSetting" min="0.15" max="0.9" step="0.05" value="0.45">
            <div style="font-size:12px;color:var(--muted);margin-top:6px">实时检测的置信度立即生效；单图/批量检测使用该默认值</div>
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
<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
function toast(msg){ const t=$('#toast'); t.textContent=msg; t.classList.add('show');
  clearTimeout(t._tm); t._tm=setTimeout(()=>t.classList.remove('show'),2200); }

/* ---------- 页面导航 ---------- */
$('#nav').addEventListener('click', e=>{
  const a = e.target.closest('a'); if(!a) return; e.preventDefault();
  $$('#nav a').forEach(x=>x.classList.remove('active')); a.classList.add('active');
  $$('.page').forEach(p=>p.classList.remove('active'));
  $('#page-'+a.dataset.page).classList.add('active');
  if(a.dataset.page==='records') loadRecords();
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
async function pollCamera(){
  try{
    const j = await (await fetch('/api/camera/status')).json();
    $('#camFps').textContent = j.on ? (j.fps+' fps') : '— fps';
    if(j.on){ $('#stVerdict').textContent = j.verdict; $('#stVerdict').style.color = j.verdict==='NG'?'var(--ng)':'var(--ok)';
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
    <div style="font-size:13px;color:var(--muted)">推理耗时 ${j.ms} ms · 置信度阈值 ${j.conf}</div>
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
    <img src="data:image/jpeg;base64,${r.image}">
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
async function loadRecords(){
  const rows=await (await fetch('/api/records')).json();
  const flt=$('#rcFilter').value;
  const shown=flt?rows.filter(r=>r.source===flt):rows;
  const ok=rows.filter(r=>r.verdict==='OK').length, ng=rows.length-ok;
  $('#rcTotal').textContent=rows.length; $('#rcOk').textContent=ok; $('#rcNg').textContent=ng;
  $('#rcRate').textContent=rows.length?Math.round(ng/rows.length*100)+'%':'—';
  $('#rcBody').innerHTML=shown.length?shown.slice(0,80).map(r=>`<tr>
    <td>${r.date}</td><td>${r.time}</td><td>${{camera:'📷 摄像头',upload:'🖼️ 单图',batch:'▦ 批量'}[r.source]||r.source}</td>
    <td class="v${r.verdict}">${r.verdict}</td><td>${r.count}</td><td>${r.detail}</td><td>${r.ms}ms</td></tr>`).join('')
    :'<tr><td colspan="7" class="empty">暂无记录</td></tr>';
}
$('#rcFilter').onchange=loadRecords;
$('#rcClear').onclick=async()=>{ if(!confirm('确认清空所有检测记录？'))return;
  await fetch('/api/records/clear',{method:'POST'}); toast('已清空'); loadRecords(); };

/* ---------- 设置 ---------- */
$('#confLiveSlider').oninput=e=>{ $('#confLive').textContent=e.target.value;
  fetch('/api/camera/conf',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({conf:parseFloat(e.target.value)})}); };
$('#confSetting').oninput=e=>$('#confSettingVal').textContent=e.target.value;
$('#saveSetting').onclick=async()=>{ const v=parseFloat($('#confSetting').value);
  await fetch('/api/camera/conf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conf:v})});
  $('#confLiveSlider').value=v; $('#confLive').textContent=v; localStorage.setItem('msConf',v); toast('设置已保存'); };
$('#resetSetting').onclick=()=>{ $('#confSetting').value=0.45; $('#confSettingVal').textContent='0.45';
  $('#confLiveSlider').value=0.45; $('#confLive').textContent='0.45'; localStorage.removeItem('msConf');
  fetch('/api/camera/conf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conf:0.45})}); toast('已恢复默认'); };
(async()=>{ const j=await (await fetch('/api/sysinfo')).json();
  $('#sysInfo').innerHTML=[['模型',j.model],['算法','YOLO26n · 100 epochs'],['mAP@0.5',j.map50],
    ['推理设备',j.device],['检测类别','scratch / dent / stain / normal_part'],['数据集',j.dataset],
    ['访问地址','http://localhost:8000']].map(x=>`<div class="kv"><span>${x[0]}</span><span>${x[1]}</span></div>`).join('');
})();
const savedConf=localStorage.getItem('msConf');
if(savedConf){ $('#confLiveSlider').value=savedConf; $('#confLive').textContent=savedConf;
  $('#confSetting').value=savedConf; $('#confSettingVal').textContent=savedConf;
  fetch('/api/camera/conf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conf:parseFloat(savedConf)})}); }
loadRecords();
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
    ann, dets, verdict, ms = detect_image(img, cam.conf)
    add_record("upload", verdict, dets, ms)
    ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"verdict": verdict, "count": len(dets), "ms": int(ms), "conf": cam.conf,
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
    results = batch_detect_images(items, cam.conf, "batch")
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
    results = batch_detect_images(items, cam.conf, "batch")
    return {"results": results, "summary": _summary(results)}


def _summary(results):
    ok = sum(1 for r in results if r["verdict"] == "OK")
    ng = len(results) - ok
    return {"total": len(results), "ok": ok, "ng": ng,
            "ngRate": f"{round(ng/len(results)*100)}%" if results else "—"}


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
    return {"model": os.path.basename(MODEL_PATH), "device": f"{DEVICE.upper()} ({GPU_NAME})",
            "map50": "0.874", "dataset": "NEU-DET + GC10-DET · 415 张"}


if __name__ == "__main__":
    print(f"MetalScan web server V2: http://localhost:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)
