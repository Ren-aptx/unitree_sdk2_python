#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLA推理客户端（完整版）

obs 组成：
  - 3路摄像头图像（head / left_wrist / right_wrist）
  - 14个手臂关节角度（弧度）
  - 14个手部关节角度（左7 + 右7，弧度）
  - 文本 prompt

数据流：
  [此脚本] --HTTP POST {prompt, images, qpos}--> [VLA服务端]
           <--{action 28维}--
  [此脚本] --> G1UpperBodyController.step(action_28)

用法：
  python vla_inference_client.py --prompt "pick up the red cup" [--network eth0] [--vla-url http://localhost:8000]
"""

import argparse
import base64
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import zmq
from g1_contorler import G1UpperBodyController

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# ────────────────────────────────────────────────────────────────
# 摄像头配置
# ────────────────────────────────────────────────────────────────
CAMERAS = {
    "head": "254322071105",  # D435i
    "left_wrist": "352122274041",  # D405
    "right_wrist": "352122273108",  # D405
}
IMAGE_W = 640
IMAGE_H = 480
CAM_FPS = 30
WARMUP_FRAMES = 30

# ────────────────────────────────────────────────────────────────
# 手臂关节名顺序（与 VLA 训练时保持一致）
# ────────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = [
    "LeftShoulderPitch",
    "LeftShoulderRoll",
    "LeftShoulderYaw",
    "LeftElbow",
    "LeftWristRoll",
    "LeftWristPitch",
    "LeftWristYaw",
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightShoulderYaw",
    "RightElbow",
    "RightWristRoll",
    "RightWristPitch",
    "RightWristYaw",
]  # 共 14 个

# G1JointIndex 中手臂关节对应的 motor_state 索引
ARM_MOTOR_INDICES = {
    "LeftShoulderPitch": 15,
    "LeftShoulderRoll": 16,
    "LeftShoulderYaw": 17,
    "LeftElbow": 18,
    "LeftWristRoll": 19,
    "LeftWristPitch": 20,
    "LeftWristYaw": 21,
    "RightShoulderPitch": 22,
    "RightShoulderRoll": 23,
    "RightShoulderYaw": 24,
    "RightElbow": 25,
    "RightWristRoll": 26,
    "RightWristPitch": 27,
    "RightWristYaw": 28,
}

# ────────────────────────────────────────────────────────────────
# 关节安全限位
# ────────────────────────────────────────────────────────────────
ARM_LIMITS = {
    "LeftShoulderPitch": (-1.6947272, 0.40309232),
    "LeftShoulderRoll": (-0.21392885, 1.69707),
    "LeftShoulderYaw": (-1.5606167, 0.63451135),
    "LeftElbow": (-1.0471996, 1.3979403),
    "LeftWristRoll": (-0.71292734, 1.9722166),
    "LeftWristPitch": (-1.6144288, 1.3940378),
    "LeftWristYaw": (-1.6144285, 1.5139462),
    "RightShoulderPitch": (-1.6966738, 0.214827),
    "RightShoulderRoll": (-1.6465511, -0.065557644),
    "RightShoulderYaw": (-0.703782, 1.4867839),
    "RightElbow": (-1.0471997, 1.4014082),
    "RightWristRoll": (-1.7988795, 0.9294281),
    "RightWristPitch": (-1.6144253, 0.83620477),
    "RightWristYaw": (-1.543896, 1.3173769),
}
LEFT_HAND_MIN = [
    -1.0471613,
    0.2968689,
    0.015484876,
    -1.8323386,
    -2.0943952,
    -1.6468145,
    -2.0943952,
]
LEFT_HAND_MAX = [0.22706091, 1.0471976, 1.6149936, 0.1919862, 0.0, 0.1919862, 0.0]
RIGHT_HAND_MIN = [-1.0471594, -1.0471976, -1.1353391, -0.1919862, 0.0, -0.1919862, 0.0]
RIGHT_HAND_MAX = [
    0.097545594,
    -0.5126544,
    -0.000534587,
    1.754668,
    2.0943952,
    1.4526877,
    2.0943952,
]

# ────────────────────────────────────────────────────────────────
# VLA 服务配置
# ────────────────────────────────────────────────────────────────
# VLA_DEFAULT_URL = "http://localhost:8000"
# VLA_INFER_ROUTE = "/infer"
ACTION_DIM = 28  # 14 arm + 7 left hand + 7 right hand


# ════════════════════════════════════════════════════════════════
class RealSenseManager:
    """多路 RealSense 摄像头管理"""

    def __init__(self, cameras: dict):
        self.cameras = cameras
        self.pipelines = {}
        self._init()

    def _init(self):
        print("[Camera] 初始化摄像头...")
        for name, serial in self.cameras.items():
            pl = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color, IMAGE_W, IMAGE_H, rs.format.bgr8, CAM_FPS
            )
            pl.start(cfg)
            for _ in range(WARMUP_FRAMES):
                pl.wait_for_frames()
            self.pipelines[name] = pl
            print(f"  ✓ {name} ({serial})")

    def capture(self) -> dict:
        """返回 {name: np.ndarray(H,W,3) BGR}"""
        out = {}
        for name, pl in self.pipelines.items():
            fs = pl.wait_for_frames()
            frame = fs.get_color_frame()
            if not frame:
                raise RuntimeError(f"[Camera] {name} 帧获取失败")
            out[name] = np.asanyarray(frame.get_data())
        return out

    def close(self):
        for name, pl in self.pipelines.items():
            pl.stop()
        print("[Camera] 全部关闭")


# ════════════════════════════════════════════════════════════════
def _img_to_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def read_arm_qpos(ctrl: G1UpperBodyController) -> np.ndarray:
    """
    从 low_state 读取 14 个手臂关节当前角度（弧度）。
    顺序与 ARM_JOINT_NAMES 一致。
    """
    low_state = ctrl.arm.low_state
    if low_state is None:
        return np.zeros(14, dtype=np.float32)
    return np.array(
        [low_state.motor_state[ARM_MOTOR_INDICES[name]].q for name in ARM_JOINT_NAMES],
        dtype=np.float32,
    )


def read_hand_qpos(ctrl: G1UpperBodyController) -> np.ndarray:
    """
    读取左手7 + 右手7 = 14 个手部关节当前角度（弧度）。
    """

    def _hand_q(hand, n=7):
        if hand.state is None:
            return np.zeros(n, dtype=np.float32)
        return np.array(
            [hand.state.motor_state[i].q for i in range(n)], dtype=np.float32
        )

    return np.concatenate([_hand_q(ctrl.left_hand), _hand_q(ctrl.right_hand)])  # (14,)


def build_payload(
    frames: dict, arm_q: np.ndarray, hand_q: np.ndarray, prompt: str
) -> dict:
    """
    组装 VLA 请求体：

    {
        "prompt": "pick up the red cup",
        "images": {
            "head":        "<base64 JPEG>",
            "left_wrist":  "<base64 JPEG>",
            "right_wrist": "<base64 JPEG>"
        },
        "qpos": {
            "arm":        [14 x float, 弧度],
            "left_hand":  [ 7 x float, 弧度],
            "right_hand": [ 7 x float, 弧度]
        }
    }

    若你的 VLA 服务端格式不同，只需修改此函数。
    """
    return {
        "prompt": prompt,
        "images": {name: _img_to_b64(img) for name, img in frames.items()},
        "qpos": {
            "arm": arm_q.tolist(),
            "left_hand": hand_q[:7].tolist(),
            "right_hand": hand_q[7:].tolist(),
        },
    }


# ════════════════════════════════════════════════════════════════
class Pi05Client:
    def __init__(self, server_ip, port=5555):

        self.ctx = zmq.Context()

        self.sock = self.ctx.socket(zmq.REQ)

        self.sock.connect(f"tcp://{server_ip}:{port}")

        print(f"Connected to {server_ip}:{port}")

    def infer(self, payload):

        self.sock.send_json(payload)

        action_bytes = self.sock.recv()

        action = np.frombuffer(action_bytes, dtype=np.float32).copy()

        if action.shape != (28,):
            raise RuntimeError(f"Bad action shape: {action.shape}")

        return action


# ════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="VLA推理客户端")
    p.add_argument(
        "--prompt", required=True, help='任务指令，例如 "pick up the red cup"'
    )
    p.add_argument("--network", default="eth0", help="网卡名（如 eth0），不填则自动")
    p.add_argument("--hz", type=float, default=10.0, help="推理频率 Hz (默认: 10)")
    p.add_argument("--max-steps", type=int, default=0, help="最大步数，0=无限")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不连接VLA，发送当前关节角回填（零动作测试）",
    )
    p.add_argument("--save-obs", action="store_true", help="保存每步图像到 ./obs_log/")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Unitree SDK 初始化 ──
    if args.network:
        ChannelFactoryInitialize(0, args.network)
    else:
        ChannelFactoryInitialize(0)

    # ── 机械臂 + 手部控制器 ──
    ctrl = G1UpperBodyController(control_dt=0.002)
    ctrl.start()  # 内部等待 LowState + HandState 就绪

    # ── 摄像头 ──
    cam = RealSenseManager(CAMERAS)

    # ── VLA 客户端 ──
    vla = None if args.dry_run else Pi05Client("192.168.123.222")

    if args.save_obs:
        os.makedirs("obs_log", exist_ok=True)

    period = 1.0 / args.hz
    step = 0

    print(f"\n{'=' * 56}")
    print(f"  任务指令 : {args.prompt}")
    print(f"  推理频率 : {args.hz} Hz  (周期 {period * 1000:.0f} ms)")
    print(
        f"  模式     : {'DRY-RUN（零动作，用于测试）' if args.dry_run else '正常推理'}"
    )
    print(f"  最大步数 : {'无限' if args.max_steps == 0 else args.max_steps}")
    print(f"{'=' * 56}")
    print("按 Ctrl+C 停止\n")

    try:
        while True:
            t0 = time.perf_counter()

            # ── 1. 读取当前关节角度 ──
            arm_q = read_arm_qpos(ctrl)  # (14,)
            hand_q = read_hand_qpos(ctrl)  # (14,) 左7+右7

            # ── 2. 采集图像 ──
            frames = cam.capture()

            # ── 3. 保存 obs（可选）──
            if args.save_obs:
                for name, img in frames.items():
                    cv2.imwrite(f"obs_log/step{step:06d}_{name}.jpg", img)

            # ── 4. 组装 payload ──
            payload = build_payload(frames, arm_q, hand_q, args.prompt)

            # ── 5. VLA 推理 → action ──
            if args.dry_run:
                # 原值回填：机械臂保持不动，用于验证 obs 采集和通路
                action = np.concatenate([arm_q, hand_q]).astype(np.float32)
            else:
                try:
                    action = vla.infer(payload)
                except Exception as e:
                    print(f"[WARN] step={step} 推理失败: {e}，跳过本帧")
                    time.sleep(period)
                    continue

            # ── 6. 关节限位保护 ──
            action = _safe_clip(action)

            # ── 7. 发送给机械臂 + 手部 ──
            ctrl.step(action)

            step += 1
            elapsed = time.perf_counter() - t0
            print(
                f"[step {step:05d}] {elapsed * 1000:5.1f}ms | "
                f"arm[:4]={action[:4].round(3).tolist()} | "
                f"lhand[:3]={action[14:17].round(3).tolist()} | "
                f"rhand[:3]={action[21:24].round(3).tolist()}"
            )

            if args.max_steps > 0 and step >= args.max_steps:
                print(f"\n已完成 {step} 步，退出。")
                break

            sleep_t = period - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n\n用户中断，停止中...")
    finally:
        ctrl.stop()
        cam.close()
        print("完成！")


if __name__ == "__main__":
    main()
