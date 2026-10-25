"""
Мост между C++ приложением и облаком LinQ TPU.

Облако LinQ (см. example.ipynb организаторов): задачи идут в RabbitMQ (брокер 10.0.114.30:5672,
exchange "tpu.direct", routing key "input"), тело — protobuf из tpu_pycloud_format с тензорами
float32, программа задаётся заголовком uri=file:///auto/cache/<LOGIN>/<имя>.tpu?raw=False,
ответ приходит в указанную нами очередь.

C++ приложение отправляет на мост кадр 640x640 (JPEG, ~60 КБ) по HTTP и получает обратно JSON с
детекциями (~1 КБ). Мост готовит тензор, ходит в облако, декодирует выходы (DFL, ключевые точки, NMS).

Три режима — выбирайте по тому, что доступно по сети (см. README):

  serve  — HTTP-сервер + инференс. Запускается там, откуда виден брокер: в Jupyter (если порт
           доступен с вашей машины) или прямо на вашей машине (если брокер виден через VPN).
             python linq_server.py serve --runtime cloud --login hackathon123 \
                 --model pose=yolov8m-pose.tpu:yolov8m-pose.json --model det=yolov8m.tpu:yolov8m.json

  worker + relay — когда брокер виден с вашей машины, но гонять 9 МБ тензоров через VPN медленно:
           worker в Jupyter берёт кадры из своей очереди в том же брокере и ходит в TPU по быстрой
           внутренней сети, relay на вашей машине принимает HTTP от C++ и пересылает только JPEG.
             Jupyter:  python linq_server.py worker --login hackathon123 --model pose=... --model det=...
             у вас:    python linq_server.py relay --login hackathon123

  serve --runtime onnx — отладка всей цепочки без LinQ (ONNX Runtime на CPU):
             python linq_server.py serve --runtime onnx --model pose=../models/yolov8m-pose_raw.onnx:../models/yolov8m-pose.json

Модель: имя=ПРОГРАММА:МЕТА.json. Для облака ПРОГРАММА — имя .tpu в /auto/cache/<LOGIN>/ (или полный путь).
HTTP API для C++: GET /health, POST /infer/<model>?conf=&iou= (тело JPEG), GET /bench/<model>?iters=
"""
import argparse
import io
import json
import os
import queue
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    import cv2
except ImportError:  # в окружении Jupyter OpenCV может не быть — тогда PIL
    cv2 = None

EXCHANGE = "tpu.direct"
TPU_QUEUE = "input"
HDR_URI, HDR_ROUTES, HDR_ID = "uri", "response_routing_keys", "tpu_pycloud_message_id"


# ================================================================== AMQP RPC
class AmqpRpc:
    """Запрос-ответ через RabbitMQ. Пул соединений: у каждого своя очередь ответов."""

    def __init__(self, host, port=5672, user="guest", password="guest", exchange=EXCHANGE):
        import pika
        self.pika = pika
        self.params = pika.ConnectionParameters(host, port, credentials=pika.PlainCredentials(user, password),
                                                heartbeat=0, blocked_connection_timeout=60)
        self.exchange = exchange
        self.pool = queue.LifoQueue()

    def _new(self):
        conn = self.pika.BlockingConnection(self.params)
        ch = conn.channel()
        rq = f"safety_reply_{uuid.uuid4().hex[:12]}"
        # exclusive, а не auto_delete: auto_delete-очередь брокер удаляет после отмены consumer-а,
        # а мы переиспользуем соединение для следующих кадров. exclusive живёт, пока живо соединение
        ch.queue_declare(queue=rq, exclusive=True)
        ch.queue_bind(exchange=self.exchange, queue=rq, routing_key=rq)
        return conn, ch, rq

    def call(self, routing_key, body, headers, timeout, expiration=None):
        for attempt in range(2):  # соединение из пула могло умереть — одна повторная попытка
            try:
                c = self.pool.get_nowait()
            except queue.Empty:
                c = self._new()
            conn, ch, rq = c
            mid = uuid.uuid4().hex
            h = dict(headers)
            h[HDR_ROUTES] = json.dumps([rq])
            h[HDR_ID] = mid
            try:
                ch.basic_publish(exchange=self.exchange, routing_key=routing_key, body=body,
                                 properties=self.pika.BasicProperties(headers=h, expiration=expiration))
                deadline = time.monotonic() + timeout
                for method, props, rbody in ch.consume(rq, auto_ack=True, inactivity_timeout=0.2):
                    if method is None:
                        if time.monotonic() > deadline:
                            ch.cancel()
                            self.pool.put(c)
                            raise TimeoutError(f"нет ответа за {timeout} с (очередь {routing_key})")
                        continue
                    rid = (props.headers or {}).get(HDR_ID)
                    if isinstance(rid, bytes):
                        rid = rid.decode()
                    if rid is not None and rid != mid:
                        continue  # опоздавший ответ на прошлый запрос (после таймаута)
                    ch.cancel()
                    self.pool.put(c)
                    return rbody, props.headers or {}
            except TimeoutError:
                raise
            except Exception:  # noqa: BLE001
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
                if attempt:
                    raise


# ================================================================== рантаймы (где считается сеть)
def program_meta(path):
    """metadata.json из .tpu, если файл доступен локально (в Jupyter — да)."""
    try:
        return json.loads(zipfile.ZipFile(path).read("metadata.json"))
    except Exception:  # noqa: BLE001
        return None


def to_nchw(a, order, expected=None):
    a = np.asarray(a)
    if a.ndim != 4:
        return a
    if order == "NHWC":
        return a.transpose(0, 3, 1, 2)
    if order is None and expected and list(a.shape) != list(expected) and \
            list(a.shape) == [expected[0], expected[2], expected[3], expected[1]]:
        return a.transpose(0, 3, 1, 2)  # метаданных программы нет, но по форме видно, что выход NHWC
    return a


def worker_ms(v, roundtrip_ms):
    """inference_time из отчёта воркера LinQ -> мс. Единицы не документированы: если значение в
    секундах, то v*1000 не больше времени всего запроса; если в мс — v*1000 заведомо больше."""
    v = float(v or 0.0)
    return v * 1e3 if v * 1e3 <= roundtrip_ms * 1.1 else v


class CloudRuntime:
    """Облако LinQ: задача в RabbitMQ -> воркер с TPU -> ответ (протокол из example.ipynb)."""

    def __init__(self, program_path, meta_json, rpc, timeout):
        self.uri = f"file://{program_path}?raw=False"
        self.rpc, self.timeout = rpc, timeout
        self.in_name = meta_json.get("input", {}).get("name", "images")
        self.in_order, self.out_orders = "NCHW", {}
        self.expected = {o["name"]: o["shape"] for o in meta_json.get("outputs", [])}
        pm = program_meta(program_path)
        if pm:
            (self.in_name, d), = [kv for g in pm["inputs"].values() for kv in g.items()]
            self.in_order = "".join(d["user_order"])
            self.out_orders = {n: "".join(o["user_order"]) for g in pm["outputs"].values() for n, o in g.items()}
        self.last_worker_time = 0.0

    def run(self, x_nchw):
        from tpu_pycloud_format import PyTask, deserialize, serialize
        x = x_nchw if self.in_order == "NCHW" else np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))
        t0 = time.perf_counter()
        body, _ = self.rpc.call(TPU_QUEUE, serialize(PyTask({self.in_name: x})), {HDR_URI: self.uri}, self.timeout)
        rt_ms = (time.perf_counter() - t0) * 1e3
        data, _, worker = deserialize(body)
        if isinstance(data, os.PathLike):  # воркер может вернуть путь к pickle с выходами
            import pickle
            with open(data, "rb") as f:
                data = pickle.load(f)
        if not data or worker.get("status_code"):
            raise RuntimeError(f"воркер LinQ: status={worker.get('status_code')} {worker.get('error_message')}")
        self.last_worker_time = worker_ms(worker.get("inference_time"), rt_ms)
        return {n: to_nchw(v, self.out_orders.get(n), self.expected.get(n)) for n, v in data.items()}


class PytpuRuntime:
    """Ускоритель виден локально (/dev/tpu*): pytpu, как в tpu_framework.pdf (Quick start)."""
    _device, _lock = None, threading.Lock()

    def __init__(self, program_path, meta_json):
        import pytpu as tpu
        if PytpuRuntime._device is None:
            devs = tpu.Device.list_devices()
            if not devs:
                raise RuntimeError("TPU не найден (ls /dev/tpu*) — используйте --runtime cloud")
            PytpuRuntime._device = tpu.Device.open(devs[0]).__enter__()
        self.inference = PytpuRuntime._device.load(program_path).__enter__().inference().__enter__()
        pm = program_meta(program_path)
        (self.in_name, d), = [kv for g in pm["inputs"].values() for kv in g.items()]
        self.in_order = "".join(d["user_order"])
        self.out_orders = {n: "".join(o["user_order"]) for g in pm["outputs"].values() for n, o in g.items()}
        self.last_worker_time = 0.0

    def run(self, x_nchw):
        x = x_nchw if self.in_order == "NCHW" else np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))
        with PytpuRuntime._lock:
            t0 = time.perf_counter()
            out = self.inference.sync({self.in_name: x})
            self.last_worker_time = (time.perf_counter() - t0) * 1e3
        return {n: to_nchw(v, self.out_orders.get(n, "NCHW")) for n, v in out.items()}


class OnnxRuntime:
    """*_raw.onnx на CPU — проверка всей цепочки без LinQ."""

    def __init__(self, path, meta_json):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=ort.get_available_providers())
        self.in_name = self.sess.get_inputs()[0].name
        self.names = [o.name for o in self.sess.get_outputs()]
        self.lock = threading.Lock()
        self.last_worker_time = 0.0

    def run(self, x_nchw):
        with self.lock:
            return dict(zip(self.names, self.sess.run(None, {self.in_name: x_nchw})))


# ================================================================== декодирование (как в src/model.cpp)
def nms(boxes, scores, cls, iou_thr, max_det=300):
    order = scores.argsort()[::-1]
    off = boxes + cls[:, None].astype(np.float32) * 7680.0
    areas = (off[:, 2] - off[:, 0]) * (off[:, 3] - off[:, 1])
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        r = order[1:]
        xx1, yy1 = np.maximum(off[i, 0], off[r, 0]), np.maximum(off[i, 1], off[r, 1])
        xx2, yy2 = np.minimum(off[i, 2], off[r, 2]), np.minimum(off[i, 3], off[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = r[inter / (areas[i] + areas[r] - inter + 1e-9) < iou_thr]
    return np.array(keep, dtype=np.int64)


class Decoder:
    def __init__(self, meta):
        self.nc = len(meta["classes"])
        self.reg_max = meta.get("reg_max", 16)
        self.imgsz = meta.get("imgsz", 640)
        ks = meta.get("kpt_shape")
        self.nk = ks[0] * ks[1] if ks else 0
        self.kdim = ks[1] if ks else 3
        self.cls_sigmoid = meta.get("cls_has_sigmoid", True)

    def levels(self, outs):
        lv = {}
        for name, a in outs.items():
            a = np.asarray(a, dtype=np.float32)
            if a.ndim != 4:
                continue
            C, H = a.shape[1], a.shape[2]
            L = lv.setdefault(H, {})
            n = name.split("/")[-1].lower()
            if n.startswith("box"):
                L["box"] = a[0]
            elif n.startswith("cls"):
                L["cls"] = a[0]
            elif n.startswith("kpt"):
                L["kpt"] = a[0]
            elif C == 4 * self.reg_max and self.nc != C:
                L["box"] = a[0]
            elif self.nk and C == self.nk and self.nc != C:
                L["kpt"] = a[0]
            else:
                L["cls"] = a[0]
        return [lv[h] for h in sorted(lv, reverse=True)]

    def __call__(self, outs, conf, iou):
        boxes, scores, cls_ids, kpts = [], [], [], []
        bins = np.arange(self.reg_max, dtype=np.float32)
        for L in self.levels(outs):
            if "box" not in L or "cls" not in L:
                continue
            c = L["cls"]
            nc, H, W = c.shape
            s = self.imgsz / H
            c = c.reshape(nc, -1)
            if not self.cls_sigmoid:
                c = 1 / (1 + np.exp(-c))
            best = c.argmax(0)
            sc = c[best, np.arange(c.shape[1])]
            idx = np.nonzero(sc >= conf)[0]
            if not idx.size:
                continue
            gx, gy = (idx % W).astype(np.float32), (idx // W).astype(np.float32)
            d = L["box"].reshape(4, self.reg_max, -1)[:, :, idx]
            d = np.exp(d - d.max(1, keepdims=True))
            d = (d / d.sum(1, keepdims=True) * bins[None, :, None]).sum(1)
            cx, cy = gx + 0.5, gy + 0.5
            boxes.append(np.stack([cx - d[0], cy - d[1], cx + d[2], cy + d[3]], 1) * s)
            scores.append(sc[idx])
            cls_ids.append(best[idx])
            if "kpt" in L and self.nk:
                k = L["kpt"].reshape(self.nk // self.kdim, self.kdim, -1)[:, :, idx]
                kx, ky = (k[:, 0] * 2 + gx) * s, (k[:, 1] * 2 + gy) * s
                kv = 1 / (1 + np.exp(-k[:, 2])) if self.kdim > 2 else np.ones_like(kx)
                kpts.append(np.stack([kx, ky, kv], -1).transpose(1, 0, 2))
        if not boxes:
            return []
        boxes, scores, cls_ids = np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls_ids)
        kpts = np.concatenate(kpts) if kpts else None
        res = []
        for i in nms(boxes, scores, cls_ids, iou):
            d = {"b": [round(float(v), 1) for v in boxes[i]], "s": round(float(scores[i]), 4), "c": int(cls_ids[i])}
            if kpts is not None:
                d["k"] = [round(float(v), 3 if j % 3 == 2 else 1) for j, v in enumerate(kpts[i].reshape(-1))]
            res.append(d)
        return res


# ================================================================== модель и локальный бэкенд
class Model:
    def __init__(self, path, meta_path, runtime, rpc=None, timeout=10.0):
        self.meta = json.load(open(meta_path, encoding="utf-8"))
        if runtime == "cloud":
            self.rt = CloudRuntime(path, self.meta, rpc, timeout)
        elif runtime == "pytpu":
            self.rt = PytpuRuntime(path, self.meta)
        else:
            self.rt = OnnxRuntime(path, self.meta)
        self.decode = Decoder(self.meta)
        self.imgsz = self.meta.get("imgsz", 640)

    def to_tensor(self, jpeg_bytes):
        if cv2 is not None:
            img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        else:
            from PIL import Image
            img = np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
        if img.shape[0] != self.imgsz or img.shape[1] != self.imgsz:
            raise ValueError(f"ожидается кадр {self.imgsz}x{self.imgsz}, пришёл {img.shape[1]}x{img.shape[0]}")
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

    def infer(self, jpeg_bytes, conf, iou):
        t0 = time.perf_counter()
        x = self.to_tensor(jpeg_bytes)
        t1 = time.perf_counter()
        outs = self.rt.run(x)
        t2 = time.perf_counter()
        dets = self.decode(outs, conf, iou)
        t3 = time.perf_counter()
        # infer — весь вызов рантайма (для облака: очередь + сеть до TPU), tpu — время самого ускорителя
        return dets, {"pre": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3,
                      "tpu": self.rt.last_worker_time}

    def bench(self, iters):
        x = np.random.default_rng(0).random((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(2):
            self.rt.run(x)
        ts, ws = [], []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.rt.run(x)
            ts.append((time.perf_counter() - t0) * 1e3)
            ws.append(self.rt.last_worker_time)
        ts.sort()
        return {"iters": iters, "mean_ms": sum(ts) / len(ts), "p50_ms": ts[len(ts) // 2],
                "p95_ms": ts[min(len(ts) - 1, int(0.95 * len(ts)))], "fps": 1000 * len(ts) / sum(ts),
                "tpu_ms_mean": sum(ws) / len(ws), "tpu_ms_p50": sorted(ws)[len(ws) // 2]}


class LocalBackend:
    def __init__(self, models, runtime):
        self.models, self.runtime = models, runtime

    def health(self):
        return {"mode": "serve", "runtime": self.runtime,
                "models": {k: {"name": m.meta["name"], "kind": m.meta["kind"], "imgsz": m.imgsz} for k, m in self.models.items()}}

    def infer(self, model, body, conf, iou):
        if model not in self.models:
            raise KeyError(f"нет модели {model}")
        dets, t = self.models[model].infer(body, conf, iou)
        return {"dets": dets, "t": t}

    def bench(self, model, iters):
        r = self.models[model].bench(iters)
        r.update(model=model, runtime=self.runtime)
        return r


class RelayBackend:
    """На вашей машине: пересылает JPEG воркеру в Jupyter через брокер LinQ."""

    def __init__(self, rpc, queue_name, timeout):
        self.rpc, self.q, self.timeout = rpc, queue_name, timeout

    def _call(self, op, body=b"", timeout=None, **kw):
        headers = {"safety_op": op, **{k: str(v) for k, v in kw.items()}}
        timeout = timeout or self.timeout
        # expiration: кадр, который воркер не успел взять за timeout, брокер выбросит сам
        try:
            rbody, _ = self.rpc.call(self.q, body, headers, timeout, expiration=str(int(timeout * 1000)))
        except TimeoutError as e:
            raise TimeoutError(f"{e}. Запущен ли worker в Jupyter с тем же --login?") from None
        r = json.loads(rbody)
        if "error" in r:
            raise RuntimeError(r["error"])
        return r

    def health(self):
        r = self._call("health")
        r["relay_queue"] = self.q
        return r

    def infer(self, model, body, conf, iou):
        return self._call("infer", body, model=model, conf=conf, iou=iou)

    def bench(self, model, iters):
        return self._call("bench", timeout=600, model=model, iters=iters)


# ================================================================== HTTP для C++
BACKEND = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            if parts and parts[-1] == "health":
                return self.reply(200, BACKEND.health())
            if len(parts) >= 2 and parts[-2] == "bench":
                return self.reply(200, BACKEND.bench(parts[-1], int(parse_qs(u.query).get("iters", ["50"])[0])))
            self.reply(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self.reply(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if len(parts) < 2 or parts[-2] != "infer":
            return self.reply(404, {"error": "not found"})
        q = parse_qs(u.query)
        try:
            self.reply(200, BACKEND.infer(parts[-1], body, float(q.get("conf", ["0.3"])[0]), float(q.get("iou", ["0.5"])[0])))
        except Exception as e:  # noqa: BLE001
            self.reply(500, {"error": str(e)})


def serve_http(backend, host, port, background=False):
    global BACKEND
    BACKEND = backend
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[linq_server] HTTP {host}:{port}", flush=True)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    srv.serve_forever()


# ================================================================== worker (в Jupyter, режим relay)
def run_worker(backend, rpc_params, queue_name, threads=4):
    """Берёт задачи из очереди queue_name брокера LinQ, отвечает JSON-ом в очередь отправителя."""
    import pika

    def loop():
        while True:  # обрыв связи с брокером не должен останавливать worker
            try:
                consume()
            except Exception as e:  # noqa: BLE001
                print(f"[linq_server] worker: {e!r}; переподключение через 3 с", flush=True)
                time.sleep(3)

    def consume():
        conn = pika.BlockingConnection(rpc_params)
        ch = conn.channel()
        ch.queue_declare(queue=queue_name, auto_delete=True)  # не оставляем мусор в брокере организаторов
        ch.queue_bind(exchange=EXCHANGE, queue=queue_name, routing_key=queue_name)
        ch.basic_qos(prefetch_count=1)

        def on_msg(chan, method, props, body):
            h = props.headers or {}
            try:
                op = h.get("safety_op", "infer")
                if op == "health":
                    res = backend.health()
                elif op == "bench":
                    res = backend.bench(h["model"], int(h.get("iters", 50)))
                else:
                    res = backend.infer(h["model"], body, float(h.get("conf", 0.3)), float(h.get("iou", 0.5)))
            except Exception as e:  # noqa: BLE001
                res = {"error": str(e)}
            routes = json.loads(h.get(HDR_ROUTES, "[]"))
            if routes:
                chan.basic_publish(exchange=EXCHANGE, routing_key=routes[-1], body=json.dumps(res).encode(),
                                   properties=pika.BasicProperties(headers={HDR_ID: h.get(HDR_ID, "")}))
            chan.basic_ack(method.delivery_tag)

        ch.basic_consume(queue=queue_name, on_message_callback=on_msg)
        ch.start_consuming()

    for _ in range(threads):
        threading.Thread(target=loop, daemon=True).start()
    print(f"[linq_server] worker слушает очередь {queue_name} ({threads} потоков)", flush=True)


# ================================================================== запуск
def build_models(specs, runtime, login, rpc, timeout):
    models = {}
    for spec in specs or []:
        name, rest = spec.split("=", 1)
        path, meta = rest.rsplit(":", 1)
        if runtime == "cloud" and not path.startswith("/"):
            path = os.path.join("/auto/cache", login, path)
        models[name] = Model(path, meta, runtime, rpc, timeout)
        print(f"[linq_server] {name}: {path} ({runtime})", flush=True)
    return models


def start(mode="serve", runtime="cloud", models=None, login="", broker="10.0.114.30", amqp_port=5672,
          user="guest", password="guest", port=8500, host="0.0.0.0", queue_name=None, timeout=10.0,
          threads=4, background=True):
    """То же из ноутбука Jupyter: linq_server.start('worker', login='hackathon123', models=[...])"""
    queue_name = queue_name or f"safety.{login or 'default'}"
    rpc = AmqpRpc(broker, amqp_port, user, password) if (runtime == "cloud" or mode == "relay") else None
    if mode == "relay":
        return serve_http(RelayBackend(rpc, queue_name, timeout), host, port, background)
    backend = LocalBackend(build_models(models, runtime, login, rpc, timeout), runtime)
    if mode == "worker":
        import pika
        run_worker(backend, pika.ConnectionParameters(broker, amqp_port, credentials=pika.PlainCredentials(user, password),
                                                       heartbeat=0), queue_name, threads)
        if not background:
            threading.Event().wait()
        return backend
    return serve_http(backend, host, port, background)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["serve", "worker", "relay"])
    ap.add_argument("--runtime", choices=["cloud", "pytpu", "onnx"], default="cloud")
    ap.add_argument("--model", action="append", help="имя=программа.tpu:модель.json (для onnx: имя=x_raw.onnx:x.json)")
    ap.add_argument("--login", default=os.environ.get("LINQ_LOGIN", ""), help="логин Jupyter (hackathonXXXXX)")
    ap.add_argument("--broker", default="10.0.114.30")
    ap.add_argument("--amqp-port", type=int, default=5672)
    ap.add_argument("--user", default="guest")
    ap.add_argument("--password", default="guest")
    ap.add_argument("--queue", help="очередь между relay и worker (по умолчанию safety.<login>)")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.mode != "relay" and not a.model:
        ap.error("укажите хотя бы одну --model")
    if a.runtime == "cloud" and a.mode != "relay" and not a.login:
        ap.error("для облака нужен --login (программы лежат в /auto/cache/<login>/)")
    start(a.mode, a.runtime, a.model, a.login, a.broker, a.amqp_port, a.user, a.password, a.port, a.host,
          a.queue, a.timeout, a.threads, background=False)
