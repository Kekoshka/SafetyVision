"""
Экспорт моделей YOLO (detect / pose / YOLO-World) в ONNX для C++ сервиса и компилятора LinQ TPU.

Почему «сырые» головы (raw):
  LinQ квантует каждый вход/выход в int8 с ОДНИМ масштабом на тензор (см. metadata.json
  внутри program.tpu: "scales": [..], "scale_shape": [1]). Стандартный выход YOLO
  (1, 4+nc, 8400) смешивает координаты 0..640 и вероятности 0..1 — после квантования
  вероятности превращаются в 0. Поэтому граф обрезается до декодирования:
      box_pX  (1, 4*reg_max, h, w)  — распределения DFL
      cls_pX  (1, nc, h, w)         — вероятности классов (sigmoid уже в графе)
      kpt_pX  (1, 17*3, h, w)       — сырые смещения ключевых точек (только pose)
  для X = 3, 4, 5. Декодирование DFL, ключевых точек и NMS делает C++ сервис.

Примеры:
  python export_models.py --models yolov8m-pose.pt yolov8m.pt --out ../models
  python export_models.py --models yolov8s-worldv2.pt \
      --world-classes "person,forklift,cell phone,cigarette,fire,smoke" --name world --out ../models
  python export_models.py --calib-src ../videos --calib-n 300 --out ../models   # только калибровка

Требования: pip install ultralytics onnx onnxslim onnxruntime opencv-python
(для YOLO-World ещё: pip install git+https://github.com/ultralytics/CLIP.git)
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mkv", ".mov", ".ts"}


class RawHead(nn.Module):
    """YOLOv8/11 detect | pose | world без декодирования: отдаёт выходы свёрток головы."""

    def __init__(self, det_model):
        super().__init__()
        from ultralytics.nn.modules.head import Pose, WorldDetect
        self.layers = det_model.model
        self.save = det_model.save
        head = self.layers[-1]
        self.is_world = isinstance(head, WorldDetect)
        self.is_pose = isinstance(head, Pose)
        if self.is_world:
            self.register_buffer("txt", det_model.txt_feats.detach().clone().float())

    def forward(self, x):
        from ultralytics.nn.modules import C2fAttn, ImagePoolingAttn
        txt = self.txt.expand(x.shape[0], -1, -1) if self.is_world else None
        ori_txt = txt
        y = []
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if self.is_world and isinstance(m, C2fAttn):
                x = m(x, txt)
            elif self.is_world and isinstance(m, ImagePoolingAttn):
                txt = m(x, txt)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
        det = self.layers[-1]
        feats = [y[j] if j != -1 else x for j in det.f]
        outs = []
        for i in range(det.nl):
            f = feats[i]
            box = det.cv2[i](f)
            cls = det.cv4[i](det.cv3[i](f), ori_txt) if self.is_world else det.cv3[i](f)
            outs += [box, cls.sigmoid()]
            if self.is_pose:
                outs.append(det.cv4[i](f))
        return tuple(outs)


def letterbox(img, size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def slim(path: Path):
    try:
        import onnx
        import onnxslim
        onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))
    except Exception as e:  # noqa: BLE001
        print(f"   [warn] onnxslim пропущен: {e}")


def export_one(weights, out_dir: Path, imgsz, opset, name=None, world_classes=None):
    from ultralytics import YOLO
    yolo = YOLO(weights)
    if world_classes:
        yolo.set_classes(world_classes)
    model = yolo.model.float().eval()
    head = model.model[-1]
    wrapper = RawHead(model).eval()
    if not wrapper.is_world:
        model.fuse()
        wrapper = RawHead(model).eval()
    kind = "pose" if wrapper.is_pose else "detect"
    name = name or Path(str(weights)).stem
    names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))
    classes = [names[i] for i in range(len(names))]

    out_names = []
    levels = ("p3", "p4", "p5", "p6")[:head.nl]
    for lvl in levels:
        out_names += [f"box_{lvl}", f"cls_{lvl}"] + ([f"kpt_{lvl}"] if kind == "pose" else [])

    onnx_path = out_dir / f"{name}_raw.onnx"
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        ref = wrapper(dummy)
    torch.onnx.export(wrapper, dummy, str(onnx_path), opset_version=opset, input_names=["images"],
                      output_names=out_names, do_constant_folding=True, dynamo=False)
    slim(onnx_path)

    # сверка ONNX Runtime с PyTorch
    import onnxruntime as ort
    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o = sess.run(None, {"images": x})
    with torch.no_grad():
        t = wrapper(torch.from_numpy(x))
    diff = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(o, t))

    meta = {
        "name": name, "kind": kind, "classes": classes, "imgsz": imgsz,
        "strides": [int(s) for s in head.stride.tolist()], "reg_max": int(head.reg_max),
        "kpt_shape": list(head.kpt_shape) if kind == "pose" else None,
        "cls_has_sigmoid": True, "input": {"name": "images", "layout": "NCHW", "color": "RGB",
                                           "scale": "1/255", "letterbox_pad": 114},
        "outputs": [{"name": n, "shape": list(r.shape)} for n, r in zip(out_names, ref)],
        "onnx": onnx_path.name, "opset": opset,
        "activation": "ReLU" if any(isinstance(m, nn.ReLU) for m in model.modules()) else "SiLU",
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[{name}] {kind}, {len(classes)} классов -> {onnx_path.name}; max|ONNX-PyTorch| = {diff:.2e}")
    return meta


def collect_frames(src: Path, n: int):
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*"))
    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    vids = [p for p in files if p.suffix.lower() in VID_EXT]
    frames = []
    per_vid = max(1, (n - min(len(imgs), n)) // max(1, len(vids))) if vids else 0
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for k in np.linspace(0, total - 1, per_vid).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
    random.Random(0).shuffle(imgs)
    for p in imgs[: max(0, n - len(frames))]:
        f = cv2.imread(str(p))
        if f is not None:
            frames.append(f)
    return frames[:n]


def make_calib(src, n, imgsz, out_dir: Path, npy: bool):
    frames = collect_frames(Path(src), n)
    d = out_dir / "calib"
    d.mkdir(parents=True, exist_ok=True)
    arr = []
    for i, f in enumerate(frames):
        lb = letterbox(f, imgsz)
        cv2.imwrite(str(d / f"{i:05d}.png"), lb)
        if npy:
            arr.append(lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0)
    if npy and arr:
        np.save(out_dir / "calib_nchw_float32.npy", np.stack(arr))
    print(f"[calib] {len(frames)} кадров -> {d}" + (" (+ .npy NCHW)" if npy else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="веса .pt (yolov8m-pose.pt, yolov8m.pt, ...)")
    ap.add_argument("--name", help="имя выходных файлов (только для одной модели)")
    ap.add_argument("--world-classes", help="классы YOLO-World через запятую")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12, help="12 — совместимо с OpenCV DNN; для LinQ уточните")
    ap.add_argument("--out", default="../models")
    ap.add_argument("--calib-src", help="папка/файл с кадрами или видео ваших камер")
    ap.add_argument("--calib-n", type=int, default=300)
    ap.add_argument("--calib-npy", action="store_true", help="дополнительно один .npy (N,3,H,W) float32")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wc = [c.strip() for c in a.world_classes.split(",")] if a.world_classes else None
    for w in a.models:
        export_one(w, out, a.imgsz, a.opset, a.name if len(a.models) == 1 else None, wc)
    if a.calib_src:
        make_calib(a.calib_src, a.calib_n, a.imgsz, out, a.calib_npy)
