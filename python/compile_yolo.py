"""
Компиляция YOLO (*_raw.onnx из export_models.py) в программу LinQ TPU (.tpu).

Основано на compile.py из SDK организаторов, отличия:
  * калибровка на РЕАЛЬНЫХ кадрах (models/calib/*.png после letterbox), а не на случайном шуме —
    для детектора это критично: пороги квантования считаются по настоящим активациям;
  * несколько моделей за один запуск, имя результата = имя модели (yolov8m-pose.tpu).

Запуск в окружении SDK (Python 3.12, пакеты из packets/requirements.txt):
    python compile_yolo.py ../models/yolov8m-pose_raw.onnx ../models/yolov8m_raw.onnx \
        --calib ../models/calib --n 64 --preset O1

Если не компилируется с O5 — используйте O1 (так советует SDK). Если после квантования
теряется точность — попробуйте --percentile 99.99 (обрезает выбросы активаций).
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx


def load_calib(calib_dir, n, shape):
    from PIL import Image
    _, c, h, w = shape
    files = sorted(glob.glob(os.path.join(calib_dir, "*.png")) + glob.glob(os.path.join(calib_dir, "*.jpg")))[:n]
    if not files:
        sys.exit(f"нет калибровочных кадров в {calib_dir} — запустите export_models.py --calib-src ...")
    arr = []
    for f in files:
        img = Image.open(f).convert("RGB")
        if img.size != (w, h):
            sys.exit(f"{f}: размер {img.size}, ожидается {(w, h)} (кадры должны быть после letterbox)")
        arr.append(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    print(f"   калибровка: {len(arr)} кадров")
    return np.stack(arr).astype(np.float32)


def remove_initialized_inputs(m):
    inits = {i.name for i in m.graph.initializer}
    for inp in list(m.graph.input):
        if inp.name in inits:
            m.graph.input.remove(inp)


def compile_one(onnx_path, args):
    from tpu_framework import (O1, O5, TPU_H1M, Network, QuantizedModel, RegularModel, TpuProgram, compiler,
                               preliminary_model_conditioning)
    preset = {"O1": O1, "O5": O5}[args.preset]
    name = os.path.basename(onnx_path).replace("_raw.onnx", "").replace(".onnx", "")
    work = os.path.join(args.work, name)
    os.makedirs(work, exist_ok=True)

    print(f"\n[{name}] 1/3 оптимизация")
    m = onnx.load(onnx_path)
    remove_initialized_inputs(m)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
    for n, s in shapes.items():
        if 0 in s:
            sys.exit(f"вход {n} имеет динамическую форму {s}: экспортируйте со статическим imgsz")
    opt = preliminary_model_conditioning(m)
    opt_path = os.path.join(work, name + "_opt.onnx")
    onnx.save(opt, opt_path)

    print(f"[{name}] 2/3 квантование")
    opt = onnx.load(opt_path)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in opt.graph.input}
    (in_name, in_shape), = shapes.items()
    calib = {in_name: load_calib(args.calib, args.n, in_shape)}
    regular = RegularModel(model=opt)
    kw = {"activation_percentile": args.percentile} if args.percentile else {}
    thresholds = regular.calibrate_torch(calibration_data=calib, **kw)
    RegularModel.dump_thresholds(os.path.join(work, "thresholds.json"), thresholds)
    quant = regular.quantize(thresholds)
    qm_name = "quant_" + name + ".qm.onnx"
    quant.save(file_dir=work, file_name=qm_name)

    print(f"[{name}] 3/3 компиляция ({args.preset})")
    qm = QuantizedModel.load(os.path.join(work, qm_name))
    network, _ = Network.from_quantized_model(qm)
    executable, descriptions = compiler.compile_(hardware_parameters=TPU_H1M, network=network, parameters=preset)
    out = os.path.join(args.out, name + ".tpu")
    TpuProgram.from_executable(executable, descriptions).to_file(out)
    print(f"[{name}] готово -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+", help="*_raw.onnx из export_models.py")
    ap.add_argument("--calib", default="../models/calib")
    ap.add_argument("--n", type=int, default=64, help="сколько калибровочных кадров (32–200)")
    ap.add_argument("--preset", choices=["O1", "O5"], default="O1")
    ap.add_argument("--percentile", type=float, default=None, help="напр. 99.99 — обрезать выбросы активаций")
    ap.add_argument("--out", default="../models")
    ap.add_argument("--work", default="../build_tpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in a.onnx:
        compile_one(p, a)
