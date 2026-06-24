#!/usr/bin/env python3
import time
import sys
import zmq
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from example.g1.low_level.g1_arm_low_level_contorl import ArmLowLevelController
from example.g1.dex.g1_hands import Dex3HandController


def hand_mode_byte(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
    mode = 0
    mode |= (motor_id & 0x0F)
    mode |= (status   & 0x07) << 4
    mode |= (timeout  & 0x01) << 7
    return mode


class G1UpperBodyController:

    ARM_NAMES = [
        "LeftShoulderPitch",  "LeftShoulderRoll",  "LeftShoulderYaw",
        "LeftElbow",          "LeftWristRoll",      "LeftWristPitch",   "LeftWristYaw",
        "RightShoulderPitch", "RightShoulderRoll",  "RightShoulderYaw",
        "RightElbow",         "RightWristRoll",     "RightWristPitch",  "RightWristYaw",
    ]

    ARM_LIMITS = {
        "LeftShoulderPitch":  (-1.6947272,  0.40309232),
        "LeftShoulderRoll":   (-0.21392885, 1.69707),
        "LeftShoulderYaw":    (-1.5606167,  0.63451135),
        "LeftElbow":          (-1.0471996,  1.3979403),
        "LeftWristRoll":      (-0.71292734, 1.9722166),
        "LeftWristPitch":     (-1.6144288,  1.3940378),
        "LeftWristYaw":       (-1.6144285,  1.5139462),
        "RightShoulderPitch": (-1.6966738,  0.214827),
        "RightShoulderRoll":  (-1.6465511, -0.065557644),
        "RightShoulderYaw":   (-0.703782,   1.4867839),
        "RightElbow":         (-1.0471997,  1.4014082),
        "RightWristRoll":     (-1.7988795,  0.9294281),
        "RightWristPitch":    (-1.6144253,  0.83620477),
        "RightWristYaw":      (-1.543896,   1.3173769),
    }

    LEFT_HAND_MIN  = [-1.0471613,  0.2968689,  0.015484876, -1.8323386, -2.0943952, -1.6468145, -2.0943952]
    LEFT_HAND_MAX  = [ 0.22706091, 1.0471976,  1.6149936,    0.1919862,  0.0,        0.1919862,  0.0]
    RIGHT_HAND_MIN = [-1.0471594, -1.0471976, -1.1353391,   -0.1919862,  0.0,       -0.1919862,  0.0]
    RIGHT_HAND_MAX = [ 0.097545594,-0.5126544,-0.000534587,  1.754668,   2.0943952,  1.4526877,  2.0943952]

    def __init__(self, control_dt: float = 0.002):
        self.arm = ArmLowLevelController(
            control_dt=control_dt,
            transition_time=0.05
        )
        self.left_hand  = Dex3HandController("L")
        self.right_hand = Dex3HandController("R")
        self.started = False

    @staticmethod
    def denormalize(x, low, high):
        x = np.clip(x, -1.0, 1.0)
        return low + (x + 1.0) * 0.5 * (high - low)

    def start(self):
        if self.started:
            return
        print("Starting Arm Controller...")
        self.arm.start()
        print("Waiting Left Hand...")
        while self.left_hand.state is None:
            time.sleep(0.05)
        print("Waiting Right Hand...")
        while self.right_hand.state is None:
            time.sleep(0.05)
        self.started = True
        print("G1UpperBodyController Ready")

    def stop(self):
        self.arm.stop()
        # 发送 timeout=1 停止帧
        for hand in (self.left_hand, self.right_hand):
            cmd = hand._new_cmd()
            for i in range(7):
                cmd.motor_cmd[i].mode = hand_mode_byte(i, status=0x01, timeout=1)
                cmd.motor_cmd[i].q    = 0.0
                cmd.motor_cmd[i].dq   = 0.0
                cmd.motor_cmd[i].kp   = 0.0
                cmd.motor_cmd[i].kd   = 0.0
                cmd.motor_cmd[i].tau  = 0.0
            hand.handcmd_pub_.Write(cmd)
        self.started = False

    def send_left_hand(self, q):
        cmd = self.left_hand._new_cmd()
        for i in range(7):
            cmd.motor_cmd[i].mode = hand_mode_byte(i, status=0x01, timeout=0)  # 注意：status 不是 statues
            cmd.motor_cmd[i].q    = float(q[i])
            cmd.motor_cmd[i].dq   = 0.0
            cmd.motor_cmd[i].kp   = 1.5
            cmd.motor_cmd[i].kd   = 0.1
            cmd.motor_cmd[i].tau  = 0.0
        self.left_hand.handcmd_pub_.Write(cmd)

    def send_right_hand(self, q):
        cmd = self.right_hand._new_cmd()
        for i in range(7):
            cmd.motor_cmd[i].mode = hand_mode_byte(i, status=0x01, timeout=0)
            cmd.motor_cmd[i].q    = float(q[i])
            cmd.motor_cmd[i].dq   = 0.0
            cmd.motor_cmd[i].kp   = 1.5
            cmd.motor_cmd[i].kd   = 0.1
            cmd.motor_cmd[i].tau  = 0.0
        self.right_hand.handcmd_pub_.Write(cmd)

    def _apply_arm_action(self, action):
        targets = {}
        for i, name in enumerate(self.ARM_NAMES):
            low, high = self.ARM_LIMITS[name]
            targets[name] = self.denormalize(action[i], low, high)
        self.arm.set_arm_targets(targets)

    def _apply_hand_action(self, left_action, right_action):
        left_q  = [self.denormalize(left_action[i],  self.LEFT_HAND_MIN[i],  self.LEFT_HAND_MAX[i])  for i in range(7)]
        right_q = [self.denormalize(right_action[i], self.RIGHT_HAND_MIN[i], self.RIGHT_HAND_MAX[i]) for i in range(7)]
        self.send_left_hand(left_q)
        self.send_right_hand(right_q)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape[0] == 28, f"期望 28 维，实际 {action.shape[0]} 维"
        self._apply_arm_action(action[:14])
        self._apply_hand_action(action[14:21], action[21:28])

    def set_arm_joint_targets(self, targets: dict):
        self.arm.set_arm_targets(targets)

    def set_left_hand_q(self, q):
        self.send_left_hand(q)

    def set_right_hand_q(self, q):
        self.send_right_hand(q)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://localhost:5555")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 100)

    ctrl = G1UpperBodyController()
    ctrl.start()

    try:
        while True:
            try:
                data = sock.recv()
                action = np.frombuffer(data, dtype=np.float32)
                ctrl.step(action)
            except zmq.Again:
                pass  # 推理端无新数据，保持当前姿态
    except KeyboardInterrupt:
        print("\n停止中...")
    finally:
        ctrl.stop()
        print("完成！")