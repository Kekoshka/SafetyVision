"""
Проверка связи с облаком LinQ и замер скорости программы .tpu — запускайте и в Jupyter, и у себя.

  1) Виден ли брокер:                python3 linq_check.py
  2) Работает ли программа на TPU:   python3 linq_check.py --login hackathonXXXXX --program yolov8m-pose.tpu \
                                         --meta yolov8m-pose.json --image кадр.jpg
  3) Замер для жюри (N запросов):    ... --bench 100 --out bench_pose_linq-cloud_s1.json
     Файл bench_*.json положите в data/ приложения — он появится на вкладке «Производительность».

Что выводит: доступность 10.0.114.30:5672, вход в AMQP, exchange tpu.direct, затем ответ воркера:
статус, имена/формы выходов, время на TPU, полное время запроса, число найденных объектов.
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linq_server as L  # noqa: E402


def ok(msg):
    print(f"  [OK]   {msg}", flush=True)


def fail(msg, hint=""):
    print(f"  [FAIL] {msg}" + (f"\n         {hint}" if hint else ""), flush=True)


def letterbox(path, size):
    """Как в C++ (model.cpp): масштаб с сохранением пропорций, серые поля 114 по центру."""
    if L.cv2 is not None:
        img = L.cv2.imread(path)
        if img is None:
            raise SystemExit(f"не читается картинка {path}")
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nw, nh = round(w * r), round(h * r)
        img = L.cv2.resize(img, (nw, nh), interpolation=L.cv2.INTER_LINEAR)
        out = np.full((size, size, 3), 114, np.uint8)
        top, left = round((size - nh) / 2 - 0.1), round((size - nw) / 2 - 0.1)
        out[top:top + nh, left:left + nw] = img
        return L.cv2.imencode(".jpg", out, [L.cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    import io
    from PIL import Image
    img = Image.open(path).convert("RGB")
    r = min(size / img.height, size / img.width)
    nw, nh = round(img.width * r), round(img.height * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="10.0.114.30")
    ap.add_argument("--amqp-port", type=int, default=5672)
    ap.add_argument("--user", default="guest")
    ap.add_argument("--password", default="guest")
    ap.add_argument("--login", default=os.environ.get("LINQ_LOGIN", ""))
    ap.add_argument("--program", help="имя .tpu в /auto/cache/<login>/ или полный путь")
    ap.add_argument("--meta", help="<модель>.json из export_models.py (классы, imgsz)")
    ap.add_argument("--image", help="тестовый кадр (любой размер)")
    ap.add_argument("--bench", type=int, default=0, help="сделать N запросов и посчитать статистику")
    ap.add_argument("--out", help="куда сохранить замер (bench_<модель>_linq-cloud_s1.json)")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    print(f"1. Брокер {a.broker}:{a.amqp_port}")
    t0 = time.perf_counter()
    try:
        socket.create_connection((a.broker, a.amqp_port), 3).close()
        ok(f"TCP-порт открыт ({(time.perf_counter() - t0) * 1e3:.0f} мс)")
    except OSError as e:
        fail(f"нет TCP-соединения: {e}",
             "С этой машины брокер не виден. Если это ваш компьютер — включите VPN хакатона или используйте "
             "вариант В из README (serve в Jupyter + прокси).")
        return 1
    try:
        import pika
    except ImportError:
        fail("нет модуля pika", "pip install pika protobuf")
        return 1
    try:
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            a.broker, a.amqp_port, credentials=pika.PlainCredentials(a.user, a.password),
            socket_timeout=5, blocked_connection_timeout=10))
        ok("вход в AMQP")
        ch = conn.channel()
        ch.exchange_declare(L.EXCHANGE, passive=True)
        ok(f"exchange {L.EXCHANGE} существует")
        conn.close()
    except Exception as e:  # noqa: BLE001
        fail(f"AMQP: {e}")
        return 1
    if not a.program:
        print("\nБрокер доступен. Дальше: --login, --program и --meta, чтобы проверить программу на TPU.")
        return 0

    try:
        import tpu_pycloud_format  # noqa: F401
    except ImportError:
        fail("нет tpu_pycloud_format", "pip install --no-deps tpu_pycloud_format-0.0.1-py3-none-any.whl")
        return 1
    if not a.meta:
        fail("нужен --meta <модель>.json")
        return 1
    path = a.program if a.program.startswith("/") else os.path.join("/auto/cache", a.login, a.program)
    print(f"\n2. Программа {path}")
    pm = L.program_meta(path)
    if pm:
        for kind in ("inputs", "outputs"):
            for g in pm[kind].values():
                for n, d in g.items():
                    print(f"         {kind[:-1]:6s} {n}: {d.get('user_shape')} {''.join(d.get('user_order', []))}")
    elif os.path.isdir("/auto/cache"):
        fail("файла нет в /auto/cache", f"скопируйте: cp {os.path.basename(path)} /auto/cache/{a.login}/")
    rpc = L.AmqpRpc(a.broker, a.amqp_port, a.user, a.password)
    model = L.Model(path, a.meta, "cloud", rpc, a.timeout)
    rt = model.rt
    print(f"         вход сети: {rt.in_name}, {rt.in_order}")

    x = np.random.default_rng(0).random((1, 3, model.imgsz, model.imgsz), dtype=np.float32)
    t0 = time.perf_counter()
    try:
        outs = rt.run(x)
    except Exception as e:  # noqa: BLE001
        fail(f"воркер: {e}", "проверьте --login, имя программы и что .tpu лежит в /auto/cache/<login>/")
        return 1
    dt = (time.perf_counter() - t0) * 1e3
    ok(f"ответ за {dt:.0f} мс (первый запрос — с загрузкой программы), TPU {rt.last_worker_time:.1f} мс")
    for n, v in outs.items():
        print(f"         output {n}: {list(v.shape)} {v.dtype}")
    levels = model.decode.levels(outs)
    if len(levels) != 3 or not all("box" in lv and "cls" in lv for lv in levels):
        fail("выходы не похожи на raw-головы YOLO (box/cls по трём масштабам)",
             "компилируйте *_raw.onnx из export_models.py")
    else:
        ok("выходы распознаны: 3 масштаба, box/cls" + ("/kpt" if all("kpt" in lv for lv in levels) else ""))

    if a.image:
        jpg = letterbox(a.image, model.imgsz)
        dets, t = model.infer(jpg, 0.3, 0.5)
        names = model.meta["classes"]
        found = {}
        for d in dets:
            found[names[d["c"]]] = found.get(names[d["c"]], 0) + 1
        ok(f"кадр {os.path.basename(a.image)}: {len(dets)} объектов {found}; "
           f"запрос {t['infer']:.0f} мс, TPU {t['tpu']:.1f} мс, декодирование {t['post']:.1f} мс")

    if a.bench:
        print(f"\n3. Замер: {a.bench} запросов")
        total, tpu, call = [], [], []
        jpg = letterbox(a.image, model.imgsz) if a.image else None
        t_start = time.perf_counter()
        for i in range(a.bench):
            t0 = time.perf_counter()
            if jpg is not None:
                _, t = model.infer(jpg, 0.3, 0.5)
                call.append(t["infer"])
                tpu.append(t["tpu"])
            else:
                t1 = time.perf_counter()
                rt.run(x)
                call.append((time.perf_counter() - t1) * 1e3)
                tpu.append(rt.last_worker_time)
            total.append((time.perf_counter() - t0) * 1e3)
        wall = time.perf_counter() - t_start

        def st(v):
            v = sorted(v)
            return {"mean": sum(v) / len(v), "p50": v[len(v) // 2], "p95": v[min(len(v) - 1, int(0.95 * len(v)))], "min": v[0]}

        S = {"total": st(total), "tpu": st(tpu), "call": st(call)}
        print(f"         TPU (отчёт воркера):  mean {S['tpu']['mean']:.1f}  p50 {S['tpu']['p50']:.1f}  p95 {S['tpu']['p95']:.1f} мс")
        print(f"         запрос через брокер:   mean {S['call']['mean']:.1f}  p95 {S['call']['p95']:.1f} мс")
        print(f"         полный цикл с декодом: mean {S['total']['mean']:.1f} мс, {a.bench / wall:.1f} FPS в один поток")
        r = {"model": model.meta.get("kind"), "model_name": model.meta["name"], "backend": "linq-cloud (RabbitMQ)",
             "imgsz": model.imgsz, "image": a.image or "random", "iters": a.bench, "streams": 1,
             "infer_ms": S["call"], "tpu_ms": S["tpu"], "total_ms": S["total"],
             "net_ms": {k: S["call"][k] - S["tpu"][k] for k in S["call"]},
             "server": {"iters": a.bench, "mean_ms": S["call"]["mean"], "p95_ms": S["call"]["p95"], "runtime": "cloud",
                        "tpu_ms_mean": S["tpu"]["mean"], "tpu_ms_p50": S["tpu"]["p50"]},
             "throughput_fps": a.bench / wall, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
             "where": socket.gethostname()}
        out = a.out or f"bench_{model.meta['name']}_linq-cloud_s1.json"
        json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        ok(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
