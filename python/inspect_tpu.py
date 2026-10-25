"""
Показывает входы и выходы скомпилированной программы LinQ (program.tpu — это zip с metadata.json).

  python inspect_tpu.py ../models/yolov8m-pose.tpu

Что проверить после компиляции:
  * вход: user_order (NCHW или NHWC), user_shape 1x3x640x640 (или 1x640x640x3), user_dtype float32;
  * выходы: 9 тензоров у pose (box/cls/kpt на 3 уровнях) или 6 у detect, у каждого СВОЙ масштаб (scales);
  * если выход один (1, 4+nc, 8400) — экспортирован не raw-вариант, вероятности классов потеряются в int8.
"""
import json
import sys
import zipfile


def main(path):
    meta = json.loads(zipfile.ZipFile(path).read("metadata.json"))
    print(f"{path}: IP {meta.get('iva_tpu_ip_version')}, API {meta.get('iva_tpu_api_version')}")
    for kind in ("inputs", "outputs"):
        print(f"\n{kind}:")
        for _, tensors in meta.get(kind, {}).items():
            for name, t in tensors.items():
                print(f"  {name:<24} user {t['user_dtype']:<8} {''.join(t['user_order']):<5} {t['user_shape']}"
                      f"  | tpu {t['tpu_dtype']} scale={[round(s, 5) for s in t.get('scales', [])][:3]}")
    hw = meta.get("hardware_parameters", {})
    print(f"\nматричный блок {hw.get('me_sa_size')}, LUT-активации: {hw.get('lut_activation_params')}")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["program.tpu"]:
        main(p)
