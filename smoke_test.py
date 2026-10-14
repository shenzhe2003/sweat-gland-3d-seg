"""
一键冒烟测试（本机 4GB 可跑，约 1-2 分钟）
============================================
验证全链路：数据 -> 模型前向/反向 -> 滑动窗口推理 -> 指标计算。
合成数据仅验证代码正确性，不代表真实性能。

用法：
  python smoke_test.py
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"[FAIL] 退出码 {r.returncode}，耗时 {time.time()-t0:.0f}s")
        sys.exit(r.returncode)
    print(f"[ok] 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    print("=" * 60)
    print("步骤 1/3：生成合成数据")
    run([PY, "prepare_data.py", "--mode", "synth", "--out", "data/volumes"])

    print("步骤 2/3：冒烟训练（2 epoch x 3 step）")
    run([PY, "train.py", "--data-dir", "data/volumes", "--smoke"])

    print("步骤 3/3：滑动窗口推理 + 指标")
    run([PY, "evaluate.py", "--data-dir", "data/volumes",
         "--ckpt", "runs/smoke/fold0/best.pt", "--fold", "0"])

    print("\n" + "=" * 60)
    print("冒烟测试通过：数据/训练/推理/指标全链路正常。")
    print("注意：合成数据上的 Dice 接近 1.0 是过拟合小数据集的结果，无意义。")
    print("下一步：准备真实数据 -> python train.py（正式训练）")
