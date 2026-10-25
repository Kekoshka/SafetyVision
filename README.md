# SafetyVision — контроль нарушений промышленной безопасности по видео на LinQ TPU

Сервис на C++ анализирует видеопотоки с камер, находит опасное поведение людей и сразу оповещает ответственных. Приложение работает на вашей машине. Нейросети выполняются на российском ускорителе LinQ TPU в облаке организаторов: задачи идут через брокер RabbitMQ по протоколу из `example.ipynb` (пакет `tpu_pycloud_format`).

## Что распознаётся


| 07 | Падение, неподвижное лежание | корпус по скелету горизонтален (или рамка шире высоты), человек не двигается ≥ 2 с | pose |
| 02 | Опасное сближение с техникой | ноги человека ближе 0,5 его роста к движущейся технике; водитель внутри техники не считается | pose + det |
| 01 | Бег, резкие перемещения | скорость точки опоры > 1,6 роста/с в течение 1 с | pose |
| 05 | Перелезание через ограждение, нахождение под грузом / конвейером | человек внутри полигона зоны типа `fence` или `under_load` | pose |
| — | Нахождение в опасной зоне (по расписанию) | зона типа `danger`, активна в заданные часы | pose |
| 03 | Телефон при работе / перемещении | телефон у головы или кисти дольше 2 с | pose + det |
| 04 | Курение, открытый огонь | сигарета у лица / огонь, дым в кадре | YOLO-World (опционально) |
| — | Нет каски / жилета | нет СИЗ в области головы / корпуса | SH17 (опционально) |

Сценарии 06 (работа на высоте без привязи) и 08 (переноска груза) не реализованы. Мы сознательно выбрали меньше сценариев, но надёжных: по условию кейса точность важнее числа сценариев.

**Подавление повторов.** Нарушение становится событием, только если условие держится `confirm_sec` подряд, поэтому случайный кадр тревогу не вызывает. На пару «человек + тип нарушения» уходит одно оповещение, пока человек в кадре (`dedup.mode = "track"`). Событие закрывается, когда условие не выполняется `release_sec`; длительность попадает в журнал.

**Журнал.** Для каждого события сохраняются:
- камера, зона, тип, время начала и оповещения, длительность;
- кадр-доказательство с выделенным человеком;
- фрагмент из 13 кадров (3 с до события);
- статус оператора («подтверждено», «ложное», «передано мастеру») и комментарий.

Журнал экспортируется в CSV. Отметки операторов дают точность по каждому типу нарушения (вкладка «Статистика»).

## Быстрый старт на ноутбуке (без ускорителя)

```bash
# 1. зависимости
sudo apt install -y build-essential cmake libopencv-dev
pip install ultralytics onnx onnxslim onnxruntime opencv-python

# 2. модели: ONNX с «сырыми» головами (см. ниже, почему)
cd python
python export_models.py --models yolov8m-pose.pt yolov8m.pt --out ../models
cd ..

# 3. сборка
mkdir -p build && cd build && cmake .. && make -j && cd ..

# 4. видео: положите свои ролики в videos/ и пропишите их в config/config.json
./build/safety_service run --config config/config.json
# интерфейс: http://localhost:8080
```

Без ускорителя модели считаются через OpenCV DNN на CPU (`"backend": {"type": "opencv"}`). Для отладки на слабом ноутбуке подойдут `yolov8n-pose.pt` и `yolov8n.pt`, но лежащего человека n-модель видит заметно хуже, чем m.

## Запуск на LinQ TPU

**1. ONNX с «сырыми» головами (на вашей машине).** LinQ квантует каждый вход и выход в int8 с одним масштабом на весь тензор: в `metadata.json` внутри `program.tpu` у тензора одно значение `scales`. Стандартный выход YOLO `(1, 4+nc, 8400)` смешивает координаты 0–640 и вероятности 0–1, поэтому после квантования вероятности обнуляются. `export_models.py` обрезает граф до декодирования: `box_p3..p5`, `cls_p3..p5` (sigmoid внутри), `kpt_p3..p5`. SDK поступает так же: в его списке поддерживаемых моделей `yolov8n_pose` обрезан до DFL.

```bash
cd python
python export_models.py --models yolov8m-pose.pt yolov8m.pt --out ../models \
       --calib-src ../videos --calib-n 300      # калибровочные кадры с ваших камер
```

**2. Компиляция в .tpu (окружение SDK, Python 3.12).** `compile_yolo.py` повторяет `compile.py` организаторов, но калибрует на реальных кадрах из `models/calib`, а не на случайном шуме. Для детектора это критично.

```bash
python compile_yolo.py ../models/yolov8m-pose_raw.onnx ../models/yolov8m_raw.onnx \
       --calib ../models/calib --n 64 --preset O1
python inspect_tpu.py ../models/yolov8m-pose.tpu    # 9 выходов, у каждого свой scale
```

Если не компилируется — оставьте `--preset O1`. Если после квантования теряется точность — добавьте `--percentile 99.99` или увеличьте `--n`.

**3. Как устроено облако LinQ.** В Jupyter загружаются только скомпилированные программы. Инференс — это сообщение в RabbitMQ (брокер `10.0.114.30:5672`, `guest/guest`, exchange `tpu.direct`, ключ `input`). Тело — protobuf `tpu_pycloud_format` с тензорами float32, заголовок `uri=file:///auto/cache/<логин>/<программа>.tpu?raw=False` выбирает программу, ответ приходит в очередь, указанную в `response_routing_keys`. Всё это делает `linq_server.py` (класс `CloudRuntime`); C++ отправляет ему кадр JPEG и получает готовые детекции.

Тензоры тяжёлые: вход 1×3×640×640 float32 — 4,9 МБ, выходы — ещё 4–5 МБ на модель. Поэтому тензоры лучше гонять внутри сети LinQ (из Jupyter), а между вашей машиной и LinQ — только JPEG и JSON.

**4. Загрузите в Jupyter** (`linq_server.py`, `linq_check.py`, `models/yolov8m-pose.json`, `models/yolov8m.json`, программы `*.tpu`, любой кадр с камеры) и проверьте программы в терминале Jupyter:

```bash
LOGIN=hackathon93845            
cp yolov8m-pose.tpu yolov8m.tpu /auto/cache/$LOGIN/
python linq_check.py --login $LOGIN --program yolov8m-pose.tpu --meta yolov8m-pose.json --image кадр.jpg --bench 100
python linq_check.py --login $LOGIN --program yolov8m.tpu      --meta yolov8m.json      --image кадр.jpg --bench 100
```

Скрипт покажет ответ воркера, имена и формы 9 выходов, число найденных людей и время: TPU по отчёту воркера и полный запрос через брокер. Замер сохраняется в `bench_*_linq-cloud_s1.json`: скачайте его в `data/` приложения, и он появится во вкладке «Производительность».

**5. Виден ли брокер с вашей машины** (VPN хакатона включён):

```bash
pip install pika protobuf
python3 python/linq_check.py          # [OK] TCP-порт открыт / [FAIL] нет TCP-соединения
```

В терминале Jupyter (окно не закрывать):

```bash
python linq_server.py worker --login $LOGIN \
    --model pose=yolov8m-pose.tpu:yolov8m-pose.json --model det=yolov8m.tpu:yolov8m.json
```

Или в ячейке ноутбука — worker работает в фоне, пока жив kernel:

```python
import linq_server
linq_server.start("worker", login="hackathon93845",
                  models=["pose=yolov8m-pose.tpu:yolov8m-pose.json", "det=yolov8m.tpu:yolov8m.json"])
```

На вашей машине:

```bash
python3 python/linq_server.py relay --login hackathon93845     # HTTP :8500 для C++
curl localhost:8500/health                                      # отвечает worker из Jupyter
```

Relay и worker находят друг друга по очереди `safety.<логин>` в брокере LinQ, поэтому команды не мешают друг другу.

**6. Конфиг на вашей машине** (`config/config.json`) для вариантов А и Б:

```json
"backend": { "type": "remote", "url": "http://127.0.0.1:8500", "timeout_sec": 10, "pipeline": 3 }
```

`pipeline` — сколько кадров камеры одновременно идут к ускорителю. Запрос в облако занимает сотни миллисекунд. Без конвейера FPS был бы равен 1 / задержку, а с конвейером результаты всё равно обрабатываются строго по порядку кадров. Модели одного кадра опрашиваются параллельно. Локально нужны только `*.json` моделей (список классов): ONNX и `.tpu` на вашей машине не используются.

**7. Сверка и замеры скорости** (обязательное требование кейса):

```bash
# детекции через LinQ и через CPU на одном кадре должны почти совпасть
./build/safety_service detect --config config/config.json --model pose --image кадр.jpg

# полный цикл с сетью + время на стороне LinQ; --streams — параллельные запросы
./build/safety_service bench --config config/config.json --model pose --image кадр.jpg --iters 200
./build/safety_service bench --config config/config.json --model pose --image кадр.jpg --iters 200 --streams 4
```

В результате `bench`:
- `tpu_ms` — чистое время ускорителя по отчёту воркера LinQ (`inference_time`);
- `infer_ms` — запрос со стороны LinQ: брокер, передача тензоров, TPU;
- `net_ms` — путь кадра от вашей машины до LinQ и обратно;
- `server` — отдельный прогон на стороне LinQ, без вашей сети и JPEG.

Результаты сохраняются в `data/bench_*.json` и показываются во вкладке «Производительность». Там же в реальном времени видно FPS каждой камеры, задержку «кадр → событие» и разбивку pre / infer / post / сеть / TPU по моделям.

**Отладка без LinQ.** Вся цепочка проверяется на одной машине через ONNX Runtime:

```bash
python python/linq_server.py serve --runtime onnx --model pose=models/yolov8m-pose_raw.onnx:models/yolov8m-pose.json
```

**Если приложение запускается прямо на машине с ускорителем** (`/dev/tpu*` и libtpu), сервер не нужен: `cmake .. -DWITH_LINQ=ON && make`, в конфиге `"backend": {"type": "linq", "device": "/dev/tpu0"}`. Промежуточный вариант — `linq_server.py serve --runtime pytpu` на этой машине.

## Конфигурация (`config/config.json`)

- `cameras[]`:
    - `source` — файл, `rtsp://…`, номер камеры или `images:папка` (набор стоп-кадров);
    - `process_fps` — сколько кадров в секунду анализировать;
    - `realtime` — проигрывать файл в реальном темпе (`true`) или обрабатывать каждый кадр (`false`);
    - `models` — какие модели запускать на этой камере;
    - `zones` — полигоны в долях кадра: `kind`, `active_from`, `active_to`;
    - `rules` — переопределение параметров правил для этой камеры.
- `rules` — параметры правил по умолчанию; `"enabled": false` выключает правило.
- `dedup` — подавление повторов: `track` (одно событие на человека) или `cooldown`.
- `webhook` — URL, на который каждое событие отправляется POST-запросом в JSON (бот мессенджера, СКУД, EHS-система).
- `vehicle_classes` — какие классы детектора считать техникой.

## Как добавить новое нарушение

1. В `src/rules.cpp` — класс-наследник `Rule` с методом `eval()` и строка `REGISTER_RULE("my_rule", MyRule)`.
2. Название и важность — в `event_types()` там же.
3. В `config.json` → `rules` — `"my_rule": {"confirm_sec": 2}`.

Конвейер, журнал, интерфейс и подавление повторов менять не нужно.

## Структура

```
src/            C++: main, model (пре/постобработка YOLO), backend_remote, backend_linq, backend_opencv,
                tracker, rules, events (журнал + подавление повторов), camera (конвейер), web (HTTP API)
web/            интерфейс (HTML/CSS/JS без внешних зависимостей — работает в закрытом контуре)
python/         export_models.py (ONNX + калибровка), compile_yolo.py (ONNX → .tpu),
                linq_server.py (мост к облаку LinQ: serve / worker / relay), linq_check.py (проверка связи
                и замер скорости на TPU), inspect_tpu.py (проверка .tpu)
config/         config.json
third_party/    cpp-httplib, nlohmann/json (MIT)
```
