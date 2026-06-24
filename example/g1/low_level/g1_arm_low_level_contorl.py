#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于低层 rt/lowcmd 接口构建的仅控制手臂的控制器。
当高层 rt/arm_sdk 话题暂时不可用时使用。
"""

import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

G1_NUM_MOTOR = 29

Kp = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20
]

Kd = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2
]


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


_ARM_JOINT_NAMES = [
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
    "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
    "RightWristRoll", "RightWristPitch", "RightWristYaw",
    "WaistYaw", "WaistRoll", "WaistPitch",
]

_JOINT_NAME_TO_INDEX = {
    name: getattr(G1JointIndex, name) for name in _ARM_JOINT_NAMES
}


def smoothstep(ratio):
    ratio = np.clip(ratio, 0.0, 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


class ArmLowLevelController:

    def __init__(self, control_dt: float = 0.002, transition_time: float = 5.0):
        self.control_dt_ = control_dt
        self.transition_time_ = transition_time

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()

        self.mode_pr_ = 0
        self.mode_machine_ = 0
        self.update_mode_machine_ = False

        self._targets: dict[int, float] = {}
        self._start_q: dict[int, float] = {}
        self._hold_q: list[float] = [0.0] * G1_NUM_MOTOR

        self._time_ = 0.0
        self.running_ = False

        # 与 Document 6 一致：先订阅，再发布
        self.lowstate_subscriber_ = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber_.Init(self._lowstate_handler, 10)

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

    # ------------------------------------------------------------------
    def _lowstate_handler(self, msg: LowState_):
        self.low_state = msg
        if not self.update_mode_machine_:
            self.mode_machine_ = self.low_state.mode_machine
            self.mode_pr_ = self.low_state.mode_pr
            self.update_mode_machine_ = True

    # ------------------------------------------------------------------
    def set_arm_target(self, joint_name: str, angle_rad: float):
        if joint_name not in _JOINT_NAME_TO_INDEX:
            raise ValueError(f"未知关节名称: {joint_name}")
        self._targets[_JOINT_NAME_TO_INDEX[joint_name]] = float(angle_rad)

    def set_arm_targets(self, targets: dict[str, float]):
        for name, angle in targets.items():
            self.set_arm_target(name, angle)

    # ------------------------------------------------------------------
    def start(self):
        if self.running_:
            return

        # 与 Document 6 的 Start() 一致：等待第一帧 lowstate
        print("等待 LowState 数据...")
        while not self.update_mode_machine_:
            time.sleep(1)
        print("已收到 LowState，开始控制线程。")

        # 记录保持位置基准和手臂起始角
        for i in range(G1_NUM_MOTOR):
            self._hold_q[i] = self.low_state.motor_state[i].q
        for idx in self._targets:
            self._start_q[idx] = self.low_state.motor_state[idx].q

        self._time_ = 0.0
        self.running_ = True

        # 与 Document 6 一致：用 RecurrentThread 驱动，不调用 Stop()
        self.lowcmd_write_thread_ = RecurrentThread(
            interval=self.control_dt_,
            target=self._lowcmd_write,
            name="arm_ctrl"
        )
        self.lowcmd_write_thread_.Start()

    def stop(self):
        """
        RecurrentThread 没有 Stop() 方法。
        通过 running_ 标志让 _lowcmd_write 内部停止写入，
        然后额外发送一帧"保持当前位置"作为收尾。
        """
        self.running_ = False
        time.sleep(self.control_dt_ * 5)   # 等几个周期让线程完成当前帧
        self._send_hold()

    # ------------------------------------------------------------------
    def _lowcmd_write(self):
        # running_ 为 False 时静默返回，线程继续跑但不写指令
        if not self.running_ or self.low_state is None:
            return

        self._time_ += self.control_dt_
        ratio = smoothstep(np.clip(self._time_ / self.transition_time_, 0.0, 1.0))

        # 与 Document 6 一致：复用同一个 low_cmd 对象而不是每次 new
        self.low_cmd.mode_pr = self.mode_pr_
        self.low_cmd.mode_machine = self.mode_machine_

        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = Kp[i]
            self.low_cmd.motor_cmd[i].kd = Kd[i]

            if i in self._targets:
                start = self._start_q[i]
                target = self._targets[i]
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * start + ratio * target
            else:
                self.low_cmd.motor_cmd[i].q = self._hold_q[i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    # ------------------------------------------------------------------
    def _send_hold(self):
        """退出时发送一帧"保持当前位置"，避免失力跌落。"""
        if self.low_state is None:
            return
        self.low_cmd.mode_pr = self.mode_pr_
        self.low_cmd.mode_machine = self.mode_machine_
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].q = self.low_state.motor_state[i].q
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = Kp[i]
            self.low_cmd.motor_cmd[i].kd = Kd[i]
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


# ----------------------------------------------------------------------
def main():
    print("警告：请确保机器人周围没有障碍物，且处于稳定站立状态。")
    input("按 Enter 键继续...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 释放高层运动模式（与 Document 6 一致）
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    status, result = msc.CheckMode()
    while result['name']:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1)

    ctrl = ArmLowLevelController(control_dt=0.002, transition_time=5.0)

    target_angle = np.deg2rad(-45.0)
    ctrl.set_arm_targets({
        "LeftShoulderPitch":  target_angle,
        "LeftShoulderRoll":   0.0,
        "LeftShoulderYaw":    0.0,
        "LeftElbow":          0.0,
        "LeftWristRoll":      0.0,
        "LeftWristPitch":     0.0,
        "LeftWristYaw":       0.0,
        "RightShoulderPitch": target_angle,
        "RightShoulderRoll":  0.0,
        "RightShoulderYaw":   0.0,
        "RightElbow":         0.0,
        "RightWristRoll":     0.0,
        "RightWristPitch":    0.0,
        "RightWristYaw":      0.0,
        "WaistYaw":           0.0,
    })

    ctrl.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        ctrl.stop()
        print("完成！")


if __name__ == "__main__":
    main()