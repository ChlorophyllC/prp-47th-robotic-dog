import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple
from glob import glob
import matplotlib

def plot_trajectory(log_file_path: str):
    """绘制轨迹和误差图"""
    # 读取日志数据
    data = np.genfromtxt(log_file_path, delimiter=',', skip_header=1,
                        names=['time', 'x', 'y', 'heading', 'target_x', 'target_y', 'target_idx'])
    
    # 提取数据
    actual_pos = np.column_stack((data['x'], data['y']))
    target_pos = np.column_stack((data['target_x'], data['target_y']))
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 子图1: 实际轨迹与理想轨迹
    plt.subplot(2, 1, 1)
    plt.plot(actual_pos[:, 0], actual_pos[:, 1], 'b-', label='实际路径')
    plt.plot(target_pos[:, 0], target_pos[:, 1], 'r--', label='理想路径')
    
    # 标记起点和终点
    plt.scatter(actual_pos[0, 0], actual_pos[0, 1], c='green', s=100, marker='o', label='实际起点')
    plt.scatter(actual_pos[-1, 0], actual_pos[-1, 1], c='blue', s=100, marker='x', label='实际终点')
    plt.scatter(target_pos[0, 0], target_pos[0, 1], c='yellow', s=100, marker='o', label='理想起点')
    plt.scatter(target_pos[-1, 0], target_pos[-1, 1], c='red', s=100, marker='x', label='理想终点')
    
    plt.title('实际路径与理想路径对比')
    plt.xlabel('X坐标(mm)')
    plt.ylabel('Y坐标(mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 子图2: 误差分析
    plt.subplot(2, 1, 2)
    
    # 计算每个点到理想路径的距离
    errors = []
    for i in range(len(actual_pos)):
        # 找到当前目标点索引对应的理想路径点
        target_idx = int(data['target_idx'][i])
        if target_idx >= len(target_pos):
            target_idx = len(target_pos) - 1
        errors.append(np.linalg.norm(actual_pos[i] - target_pos[target_idx]))
    
    plt.plot(data['time'] - data['time'][0], errors, 'r-', label='路径误差')
    plt.title('路径误差随时间变化')
    plt.xlabel('时间 (s)')
    plt.ylabel('误差 (mm)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = log_file_path.replace('.log', '_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"轨迹和误差图已保存到: {plot_path}")

def get_latest_log_file(folder_path: str):
    """获取文件夹中最新的日志文件"""
    log_files = glob(os.path.join(folder_path, '*.log'))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {folder_path}")
    
    # 按修改时间排序
    log_files.sort(key=os.path.getmtime)
    return log_files[-1]

import time 
import socket
car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def cvt_ctrl_to_car_ctrl(speed, angle, bias=0.0):
    """
    将控制指令转换为车辆控制格式
    
    :param speed: 速度
    :param angle: 角度
    :return: 控制缓冲区
    """
    buffer = np.zeros(4)
    # 左轮右轮补偿电机性能差异
    left_speed = speed - angle - bias
    right_speed = speed + angle + bias

    buffer[0] = max(-100, min(100, left_speed))   # 左前
    buffer[1] = max(-100, min(100, right_speed))  # 右前
    buffer[2] = max(-100, min(100, left_speed))   # 左后
    buffer[3] = max(-100, min(100, right_speed))  # 右后
    return buffer
def send_ctrl(speed, angle,ip,port):
    """
    发送控制指令
    :param speed: 径向速度
    :param angle: 角速度
    :return:
    """
    buffer = cvt_ctrl_to_car_ctrl(speed, angle, bias=7)  # 将控制输入转换为小车控制输入
    command = "<%d,%d,%d,%d>" % (int(buffer[0]), int(buffer[1]), int(buffer[2]), int(buffer[3]))
    car_communication.sendto(command.encode(), (ip, port))  # 发送控制指令
            
def test_straight_line(ip,port):
    speed = 60
    angle = 0
    for i in range(30):  # 运行 3 秒左右
        send_ctrl(speed, angle, ip, port)
        time.sleep(0.1)
    send_ctrl(0, 0, ip, port)

ip_1 = '192.168.1.205'
port_1 = int(12345)

ip_2 = '192.168.1.205'
port_2 = int(12345)

ip_3 = '192.168.1.201'
port_3 = int(12345)

ip_4 = '192.168.1.208'
port_4 = int(12345)

# test_straight_line(ip_1, port_1)

# 示例使用
if __name__ == "__main__":
    matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
    # 1. 获取最新的日志文件
    log_folder = "trajectory_logs"
    latest_log = get_latest_log_file(log_folder)

    # 2. 绘制轨迹图
    plot_trajectory(latest_log)