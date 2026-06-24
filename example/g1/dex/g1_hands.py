#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dex3-1 灵巧手控制示例（Python 版本）
对应官方 C++ 示例，使用 unitree_sdk2py 中的 HandCmd_ / HandState_。
"""

import sys
import time
import threading
import select
import termios
import tty
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

MOTOR_MAX = 7
SENSOR_MAX = 9

# URDF 限位 (与 C++ 示例一致)
maxLimits_left  = [ 1.05,  1.05, 1.75,  0.0,   0.0,   0.0,   0.0]
minLimits_left  = [-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75]
maxLimits_right = [ 1.05,  0.742, 0.0,  1.57,  1.75,  1.57,  1.75]
minLimits_right = [-1.05, -1.05, -1.75, 0.0,   0.0,   0.0,   0.0]


class RIS_Mode:
    """把 (id, status, timeout) 打包成 uint8 mode 字段"""
    def __init__(self, id_=0, status=0x01, timeout=0):
        self.id = id_
        self.status = status
        self.timeout = timeout

    def to_uint8(self) -> int:
        mode = 0
        mode |= (self.id & 0x0F)
        mode |= (self.status & 0x07) << 4
        mode |= (self.timeout & 0x01) << 7
        return mode


class Dex3HandController:
    def __init__(self, hand_id: str, network_iface: str | None = None):
        """
        hand_id: 'L' 或 'R'
        network_iface: 网卡名（与示例中的 argv[1] 对应），可为 None
        """
        if hand_id not in ("L", "R"):
            raise ValueError("hand_id 必须是 'L' 或 'R'")

        self.is_left = (hand_id == "L")
        self.max_limits = maxLimits_left if self.is_left else maxLimits_right
        self.min_limits = minLimits_left if self.is_left else minLimits_right

        if self.is_left:
            cmd_topic = "rt/dex3/left/cmd"
            state_topic = "rt/dex3/left/state"
        else:
            cmd_topic = "rt/dex3/right/cmd"
            state_topic = "rt/dex3/right/state"

        if network_iface:
            ChannelFactoryInitialize(0, network_iface)
        else:
            ChannelFactoryInitialize(0)

        self.handcmd_pub_ = ChannelPublisher(cmd_topic, HandCmd_)
        self.handcmd_pub_.Init()

        self.handstate_sub_ = ChannelSubscriber(state_topic, HandState_)
        self.handstate_sub_.Init(self._state_handler, 10)

        self.state: HandState_ | None = None

        # 旋转动画用的计数器
        self._count = 1
        self._dir = 1

    # ------------------------------------------------------------------
    def _state_handler(self, msg: HandState_):
        self.state = msg

    # ------------------------------------------------------------------
    def _new_cmd(self) -> HandCmd_:
        cmd = unitree_hg_msg_dds__HandCmd_()
        # motor_cmd 默认数组长度通常已经够用；如不够则手动 resize（不同 SDK 版本可能需要）
        return cmd

    # ------------------------------------------------------------------
    def rotate_step(self):
        """对应 C++ rotateMotors：用 sin 波在限位范围内来回摆动"""
        cmd = self._new_cmd()
        for i in range(MOTOR_MAX):
            ris = RIS_Mode(id_=i, status=0x01, timeout=0)
            mc = cmd.motor_cmd[i]
            mc.mode = ris.to_uint8()
            mc.tau = 0.0
            mc.kp = 0.5
            mc.kd = 0.1

            rng = self.max_limits[i] - self.min_limits[i]
            mid = (self.max_limits[i] + self.min_limits[i]) / 2.0
            amplitude = rng / 2.0
            q = mid + amplitude * np.sin(self._count / 20000.0 * np.pi)
            mc.q = q
            mc.dq = 0.0

        self.handcmd_pub_.Write(cmd)

        self._count += self._dir
        if self._count >= 10000:
            self._dir = -1
        if self._count <= -10000:
            self._dir = 1

    # ------------------------------------------------------------------
    def grip(self):
        """对应 C++ gripHand：移动到限位中点（闭合/张开到中间位置）"""
        cmd = self._new_cmd()
        for i in range(MOTOR_MAX):
            ris = RIS_Mode(id_=i, status=0x01, timeout=0)
            mc = cmd.motor_cmd[i]
            mc.mode = ris.to_uint8()
            mc.tau = 0.0

            mid = (self.max_limits[i] + self.min_limits[i]) / 2.0
            mc.q = mid
            mc.dq = 0.0
            mc.kp = 1.5
            mc.kd = 0.1

        self.handcmd_pub_.Write(cmd)

    # ------------------------------------------------------------------
    def stop(self):
        """对应 C++ stopMotors：timeout=1，kp/kd/q 全部置 0"""
        cmd = self._new_cmd()
        for i in range(MOTOR_MAX):
            ris = RIS_Mode(id_=i, status=0x01, timeout=1)
            mc = cmd.motor_cmd[i]
            mc.mode = ris.to_uint8()
            mc.tau = 0.0
            mc.dq = 0.0
            mc.kp = 0.0
            mc.kd = 0.0
            mc.q = 0.0

        self.handcmd_pub_.Write(cmd)

    # ------------------------------------------------------------------
    def print_state(self):
        """对应 C++ printState：打印归一化后的关节位置"""
        if self.state is None:
            print("尚未收到 HandState_ 数据...")
            return

        q_norm = []
        for i in range(MOTOR_MAX):
            q = self.state.motor_state[i].q
            rng = self.max_limits[i] - self.min_limits[i]
            if rng == 0:
                q_n = 0.0
            else:
                q_n = (q - self.min_limits[i]) / rng
                q_n = float(np.clip(q_n, 0.0, 1.0))
            q_norm.append(q_n)

        side = "L" if self.is_left else "R"
        print("\033[2J\033[H", end="")  # 清屏
        print("-- Hand State --")
        print(f"{side}: " + " ".join(f"{v:.3f}" for v in q_norm))


# ----------------------------------------------------------------------
# 非阻塞键盘输入（替代 C++ 中的 termios + fcntl 实现）
# ----------------------------------------------------------------------
class NonBlockingInput:
    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)

    def __enter__(self):
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:
            return sys.stdin.read(1)
        return None


# ----------------------------------------------------------------------
# 主程序：状态机（INIT / ROTATE / GRIP / STOP / PRINT）
# ----------------------------------------------------------------------
def main():
    print(" --- Unitree Robotics --- ")
    print("     Dex3 Hand Example (Python)      \n")

    hand_id = input("请输入手的 ID (L = 左手, R = 右手): ").strip().upper()
    if hand_id not in ("L", "R"):
        print("无效的手部 ID，请输入 'L' 或 'R'。")
        sys.exit(-1)

    network_iface = sys.argv[1] if len(sys.argv) > 1 else None
    if network_iface is None:
        print(f"用法: {sys.argv[0]} <网络接口名>")
        sys.exit(-1)

    ctrl = Dex3HandController(hand_id, network_iface)

    # 等待第一帧状态
    print("等待 HandState_ 数据...")
    while ctrl.state is None:
        time.sleep(0.1)
    print("已连接。")

    state = "ROTATE"  # 对应 C++ 的 INIT -> ROTATE
    last_state = None

    def print_menu():
        print(f"\n--- Current State: {state} ---")
        print("Commands:")
        print("  r - Rotate")
        print("  g - Grip")
        print("  p - Print_state")
        print("  q - Quit")
        print("  s - Stop")

    try:
        with NonBlockingInput() as nb:
            while True:
                key = nb.get_key()
                if key == 'q':
                    print("Exiting...")
                    state = "STOP"
                    ctrl.stop()
                    break
                elif key == 'r':
                    state = "ROTATE"
                elif key == 'g':
                    state = "GRIP"
                elif key == 'p':
                    state = "PRINT"
                elif key == 's':
                    state = "STOP"

                if state != last_state:
                    print_menu()
                    last_state = state

                if state == "ROTATE":
                    ctrl.rotate_step()
                    time.sleep(0.0001)  # 对应 C++ usleep(100)
                elif state == "GRIP":
                    ctrl.grip()
                    time.sleep(1.0)     # 对应 C++ usleep(1000000)
                elif state == "STOP":
                    ctrl.stop()
                    time.sleep(1.0)
                elif state == "PRINT":
                    ctrl.print_state()
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n中断，停止电机...")
        ctrl.stop()


if __name__ == "__main__":
    main()