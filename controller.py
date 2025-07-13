import numpy as np
import math
import time
from typing import List, Tuple
import socket
from geometry_msgs.msg import PoseStamped, TwistStamped
import threading
import rospy
import os
from datetime import datetime
import matplotlib.pyplot as plt

class PID_posi:
    """
    位置式实现
    """
    def __init__(self, kp, ki, kd, target, upper=1., lower=-1.):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err = 0
        self.err_last = 0
        self.err_all = 0
        self.target = target
        self.upper = upper
        self.lower = lower
        self.value = 0

    def cal_output(self, state):
        self.err = self.target - state
        # self.err =state-self.target
        self.value = self.kp * self.err + self.ki * \
            self.err_all + self.kd * (self.err - self.err_last)
        self.update()
        return self.value

    def update(self):
        self.err_last = self.err
        self.err_all = self.err_all + self.err
        if self.value > self.upper:
            self.value = self.upper
        elif self.value < self.lower:
            self.value = self.lower

    def auto_adjust(self, Kpc, Tc):
        self.kp = Kpc * 0.6
        self.ki = self.kp / (0.5 * Tc)
        self.kd = self.kp * (0.125 * Tc)
        return self.kp, self.ki, self.kd

    def set_pid(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0

    def set_target(self, target):
        self.target = target

class TrajectoryPlanner:
    """
    轨迹跟踪规划器
    """
    def __init__(self, 
                 trajectory_points: List[Tuple[float, float]],
                 lookahead_distance: float = 3.0,
                 max_speed: float = 50.0,
                 min_speed: float = 0.0,
                 goal_tolerance: float = 1.5,
                 max_angle_control: float = 50.0,
                 bias = 0,
                 name = None):
        """
        初始化轨迹跟踪器
        
        :param trajectory_points: 轨迹点列表 [(x1,y1), (x2,y2), ...]
        :param lookahead_distance: 前瞻距离
        :param max_speed: 最大速度
        :param min_speed: 最小速度
        :param goal_tolerance: 目标点容差
        :param max_angle_control: 最大角度控制量
        """
        self.trajectory_points = self.interpolate_path(trajectory_points, spacing=500)
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.goal_tolerance = goal_tolerance
        self.max_angle_control = max_angle_control
        self.bias = bias
        self.name = name

        # 当前状态
        self.current_pos = np.array([0.0, 0.0])
        self.current_twist = None  # 用于接收速度信息
        self.trajectory_history = []
        # 初始化位置和朝向
        self.current_heading = 0.0
        self.current_target_idx = 0
        
        # 控制状态
        self.is_finished = False
        self.last_time = time.time()
        self.stuck_counter = 0
        self.last_pos = np.array([0.0, 0.0])
        self.angle_error_integral = 0.0
        
        # 通信socket
        self.car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Log 文件
        self.log_file = None
        self.create_log_file()
        
    def update_position(self, x: float, y: float, heading: float):
        """
        更新当前位置和朝向
        
        :param x: x坐标
        :param y: y坐标
        :param heading: 朝向角度（弧度）
        """
        self.current_pos = np.array([x, y])
        self.current_heading = heading
        self.trajectory_history.append((x, y))

        if self.log_file:
            target_x, target_y, target_idx = self.find_target_point()
            self.log_file.write(
                f"{time.time()},{x},{y},{heading},{target_x},{target_y},{target_idx}\n"
            )
            self.log_file.flush()
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        return self.trajectory_history.copy()   
     
    def interpolate_path(self, points, spacing=30.0):
        """
        按照路径长度插值（每隔 spacing 插一个点）

        :param points: 原始路径 [(x, y), ...]
        :param spacing: 每两个点之间的距离（单位 mm）
        :return: 插值后的路径点列表 [(x, y), ...]
        """
        interpolated = []
        for i in range(len(points) - 1):
            p0 = np.array(points[i])
            p1 = np.array(points[i + 1])
            segment_vec = p1 - p0
            segment_len = np.linalg.norm(segment_vec)

            if segment_len < 1e-6:
                continue

            direction = segment_vec / segment_len
            num_points = max(int(segment_len // spacing), 1)

            for j in range(num_points):
                new_point = p0 + direction * spacing * j
                interpolated.append(tuple(new_point))

        interpolated.append(tuple(points[-1]))  # 保证最后一个点也加入
        return interpolated
    
    def find_target_point(self) -> Tuple[float, float, int]:
        """
        寻找前瞻目标点（基于当前位置 + 向前搜索一段lookahead距离）
        :return: (target_x, target_y, target_index)
        """
        lookahead_distance = getattr(self, 'lookahead_distance', 200.0)  # 默认值
        best_idx = self.current_target_idx  # 起始点，不回退

        for i in range(self.current_target_idx, len(self.trajectory_points)):
            pt = self.trajectory_points[i]
            dist = np.linalg.norm(self.current_pos - pt)
            if dist >= lookahead_distance:
                best_idx = i
                break
        else:
            # 没找到足够远的，选最后一个点
            best_idx = len(self.trajectory_points) - 1

        self.current_target_idx = best_idx  # 向前推进
        target_point = self.trajectory_points[best_idx]
        return target_point[0], target_point[1], best_idx

    

    def normalize_angle(self, angle):
        """
        将角度规范化到[-π, π]范围内
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calculate_control(self) -> Tuple[float, float]:
        """
        计算控制指令
        :return: (speed, angle)
        """
        if self.is_finished:
            return 0.0, 0.0
        
        # 初始化控制器
        if not hasattr(self, 'angle_pid'):
            # 角度PID控制器 - 控制角速度
            self.angle_pid = PID_posi(
                kp=1.0,
                ki=0.1,
                kd=1.0,
                target=0.0,  # 目标角度误差为0
                upper=self.max_angle_control,
                lower=-self.max_angle_control
            )
            
            # 速度PID控制器 - 控制前进速度
            self.speed_pid = PID_posi(
                kp=1.5,
                ki=0.2,
                kd=0.1,
                target=0.0,  # 目标距离误差
                upper=self.max_speed,
                lower=0.0
            )
            
            # 延迟补偿和预测参数
            self.ros_delay = 0
            self.position_history = []
            self.heading_history = []
            self.control_history = []
            self.max_history_size = 25
            
            # 运动预测参数
            self.last_update_time = time.time()
            self.predicted_pos = None
            self.predicted_heading = None
            
            # 卡住检测相关参数
            self.stuck_counter = 0
            self.last_pos = self.current_pos.copy()
            self.velocity_stuck_counter = 0
            self.stuck_threshold = 2.0  # mm
            self.stuck_time_threshold = 3.0  # 秒
            
            # 电机补偿参数 - 用于处理左右电机不一致
            self.motor_bias = 0.0  # 电机偏差补偿
            self.motor_bias_history = []
            self.motor_bias_alpha = 0.95  # 滤波系数
            
            # 基础控制参数
            self.min_angle_control = 25.0
            self.speed_efficiency = 4.95
            
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 更新历史记录
        self._update_history()
        
        # 运动预测 (补偿ROS延迟)
        predicted_pos, predicted_heading = self._predict_motion(dt)
        
        # 获取目标点
        target_x, target_y, target_idx = self.find_target_point()
        
        # 检查是否到达终点
        if target_idx >= len(self.trajectory_points) - 1:
            distance_to_end = np.linalg.norm(predicted_pos - self.trajectory_points[-1])
            if distance_to_end < self.goal_tolerance:
                self.is_finished = True
                print("🎉 轨迹跟踪完成！")
                return 0.0, 0.0
        
        # 计算目标向量和误差
        target_vector = np.array([target_x, target_y]) - predicted_pos
        target_distance = np.linalg.norm(target_vector)
        target_angle = math.atan2(target_vector[1], target_vector[0])
        angle_error = -self.normalize_angle(target_angle - predicted_heading)

        
        # 卡住检测
        if self._is_stuck():
            print("⚠️ 检测到卡住，执行脱困操作")
            self.angle_pid.reset()
            self.speed_pid.reset()
            # 脱困：后退 + 转向
            return -self.min_speed * 0.5, max(25.0, 50.0 * np.sign(angle_error))
        
        # 使用PID控制器计算控制输出
        speed, angle = self._calculate_pid_control(target_distance, angle_error, dt)
        
        # 电机偏差补偿
        angle = self._apply_motor_bias_compensation(angle, angle_error)
        
        # 输出平滑
        speed, angle = self._smooth_output_with_delay_compensation(speed, angle)
        
        # 记录控制指令历史
        self.control_history.append({
            'time': current_time,
            'speed': speed,
            'angle': angle
        })
        if len(self.control_history) > self.max_history_size:
            self.control_history.pop(0)
        
        # 调试信息
        self._print_debug_info(predicted_pos, predicted_heading, target_x, target_y, 
                            target_distance, angle_error, speed, angle)
        
        return speed, angle

    def _is_stuck(self) -> bool:
        """
        检测小车是否卡住
        :return: True表示卡住，False表示正常
        """
        # 位置变化检测
        current_pos = self.current_pos
        if hasattr(self, 'last_pos'):
            position_change = np.linalg.norm(current_pos - self.last_pos)
            
            # 如果位置变化很小，增加卡住计数
            if position_change < self.stuck_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                
            # 更新上次位置
            self.last_pos = current_pos.copy()
            
            # 如果连续多次位置变化很小，认为卡住
            if self.stuck_counter > int(self.stuck_time_threshold / 0.1):  # 假设更新频率10Hz
                return True
        
        # 速度检测 - 如果发送了控制指令但速度很低
        if hasattr(self, 'control_history') and len(self.control_history) > 5:
            recent_controls = self.control_history[-5:]
            avg_speed_command = np.mean([cmd['speed'] for cmd in recent_controls])
            
            # 如果指令速度不为0但实际移动很少，可能卡住
            if avg_speed_command > 20.0 and hasattr(self, 'position_history') and len(self.position_history) > 5:
                recent_position_changes = []
                # 计算最近几次位置变化
                for i in range(1, min(len(self.position_history), 6)):
                    if len(self.position_history) > i:
                        # position_history存储的是 {'time': xxx, 'pos': [x, y]} 格式
                        pos_current = self.position_history[-1]['pos']
                        pos_previous = self.position_history[-i-1]['pos']
                        
                        change = np.linalg.norm(pos_current - pos_previous)
                        recent_position_changes.append(change)
                
                if recent_position_changes:
                    avg_position_change = np.mean(recent_position_changes)
                    if avg_position_change < 30.0:  # 期望的移动距离很小
                        self.velocity_stuck_counter += 1
                    else:
                        self.velocity_stuck_counter = 0
                        
                    if self.velocity_stuck_counter > 10:
                        return True
        
        return False
    
    def _calculate_pid_control(self, target_distance: float, angle_error: float, dt: float) -> Tuple[float, float]:
        """
        使用PID控制器计算速度和角速度
        :param target_distance: 到目标点的距离
        :param angle_error: 角度误差
        :param dt: 时间间隔
        :return: (speed, angle)
        """
        # 角度控制 - 使用PID控制器
        # 目标是让角度误差为0
        angle_output = self.angle_pid.cal_output(angle_error)
        
        # 速度控制 - 根据距离和角度误差调整
        # 期望的跟踪距离
        desired_distance = self.lookahead_distance * 0.8 if hasattr(self, 'lookahead_distance') else 200.0
        distance_error = target_distance - desired_distance
        
        # 基础速度计算
        base_speed = self._calculate_adaptive_speed(target_distance, angle_error)
        
        # 使用PID调整速度
        speed_adjustment = self.speed_pid.cal_output(distance_error)
        speed = base_speed + speed_adjustment * 0.5  # 降低PID调整幅度
        
        # 根据角度误差调整速度 - 角度误差大时减速
        angle_error_deg = abs(math.degrees(angle_error))
        if angle_error_deg > 30:
            speed *= 0.4  # 大角度误差时减速
        elif angle_error_deg > 15:
            speed *= 0.6  # 中等角度误差时适当减速
        
        # 限制输出范围
        speed = np.clip(speed, self.min_speed, self.max_speed)
        angle = np.clip(angle_output, -self.max_angle_control, self.max_angle_control)
        
        # 确保最小角速度控制值
        if abs(angle) > 0 and abs(angle) < self.min_angle_control:
            angle = self.min_angle_control * np.sign(angle)
        
        print(f"🎯 PID控制 | 角度误差: {math.degrees(angle_error):.1f}°, "
            f"距离误差: {distance_error:.1f}mm, 速度: {speed:.1f}, 角速度: {angle:.1f}")
        
        return speed, angle

    def _apply_motor_bias_compensation(self, angle: float, angle_error: float) -> float:
        """
        应用电机偏差补偿，处理左右电机不一致问题
        :param angle: 原始角速度控制值
        :param angle_error: 当前角度误差
        :return: 补偿后的角速度控制值
        """
        # 如果小车在直行时持续偏向一边，说明电机有偏差
        if abs(angle) < 5.0:  # 基本直行状态
            # 记录角度误差历史
            self.motor_bias_history.append(angle_error)
            if len(self.motor_bias_history) > 20:
                self.motor_bias_history.pop(0)
            
            # 计算平均角度偏差
            if len(self.motor_bias_history) > 10:
                avg_bias = np.mean(self.motor_bias_history)
                
                # 如果持续偏向一边，更新电机偏差补偿
                if abs(avg_bias) > math.radians(5):  # 超过5度偏差
                    # 使用指数滤波更新偏差补偿
                    bias_compensation = -avg_bias * 10.0  # 转换为控制量
                    self.motor_bias = self.motor_bias * self.motor_bias_alpha + \
                                    bias_compensation * (1 - self.motor_bias_alpha)
                    
                    # 限制补偿范围
                    self.motor_bias = np.clip(self.motor_bias, -15.0, 15.0)
        
        # 应用补偿
        compensated_angle = angle + self.motor_bias
        
        # 如果补偿后的角速度很小，设为0避免抖动
        if abs(compensated_angle) < 3.0:
            compensated_angle = 0.0
        
        print(f"🔧 电机补偿 | 原始角速度: {angle:.1f}, 偏差补偿: {self.motor_bias:.1f}, "
            f"补偿后: {compensated_angle:.1f}")
        
        return compensated_angle

    def _calculate_adaptive_speed(self, distance: float, angle_error: float) -> float:
        """
        计算自适应速度
        :param distance: 到目标点距离
        :param angle_error: 角度误差
        :return: 速度值
        """
        # 基础速度根据距离调整
        if distance > 500:
            base_speed = self.max_speed
        elif distance > 300:
            base_speed = self.max_speed * 0.8
        elif distance > 150:
            base_speed = self.max_speed * 0.6
        else:
            base_speed = self.max_speed * 0.4
        
        # 根据角度误差调整
        angle_factor = 1.0 - min(abs(angle_error) / math.pi, 0.5)
        
        return max(self.min_speed, base_speed * angle_factor)

    def _print_debug_info(self, predicted_pos, predicted_heading, target_x, target_y, 
                        target_distance, angle_error, speed, angle):
        """
        打印调试信息
        """
        angle_error_deg = math.degrees(angle_error)
        print(f"📊 状态报告:")
        print(f"   实际位置: ({self.current_pos[0]:.2f}, {self.current_pos[1]:.2f})")
        print(f"   预测位置: ({predicted_pos[0]:.2f}, {predicted_pos[1]:.2f})")
        print(f"   实际朝向: {self.current_heading * 180 / math.pi:.1f}°")
        print(f"   预测朝向: {predicted_heading * 180 / math.pi:.1f}°")
        print(f"   目标: ({target_x:.1f}, {target_y:.1f})")
        print(f"   距离: {target_distance:.2f}mm")
        print(f"   角度误差: {angle_error_deg:.1f}°")
        print(f"   控制输出: 速度={speed:.1f}, 角速度={angle:.1f}")
        print(f"   电机偏差补偿: {self.motor_bias:.2f}")
        print(f"   卡住计数: {self.stuck_counter}")
    
    def _update_history(self):
        """更新历史记录"""
        current_time = time.time()
        
        # 更新位置历史
        self.position_history.append({
            'time': current_time,
            'pos': self.current_pos.copy()
        })
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        
        # 更新朝向历史
        self.heading_history.append({
            'time': current_time,
            'heading': self.current_heading
        })
        if len(self.heading_history) > self.max_history_size:
            self.heading_history.pop(0)

    def _predict_motion(self, dt: float) -> Tuple[np.ndarray, float]:
        """
        使用控制历史与位置历史融合，预测当前位置与朝向，补偿延迟
        """
        delay = self.ros_delay

        # 获取平均线速度（基于控制历史或位置差值）
        if len(self.control_history) >= 1:
            last_cmd = self.control_history[-1]
            speed = last_cmd['speed'] / 1000.0  # mm/s → m/s
            angle_deg = last_cmd['angle']
        else:
            speed = 0.0
            angle_deg = 0.0

        # 获取当前朝向
        heading = self.current_heading
        angular_velocity = math.radians(angle_deg) / dt  # 角速度 (rad/s)

        # 朝向更新
        dtheta = angular_velocity * delay
        predicted_heading = self.normalize_angle(heading + dtheta)

        # 平移预测
        dx = speed * math.cos(heading) * delay
        dy = speed * math.sin(heading) * delay
        predicted_pos = self.current_pos + np.array([dx * 1000, dy * 1000])  # 转回 mm

        return predicted_pos, predicted_heading

    def _smooth_output_with_delay_compensation(self, speed, angle):
        """
        考虑延迟的输出平滑
        """
        if not hasattr(self, 'last_speed'):
            self.last_speed = 0.0
            self.last_angle = 0.0
        
        # 根据角度误差大小调整平滑系数
        if abs(angle) > 20.0:  # 角度控制较大时
            # 减少平滑以提高响应
            alpha = 0.9
        else:
            # 速度控制或小角度调整时，适度平滑
            alpha = 0.7
        
        speed = alpha * speed + (1 - alpha) * self.last_speed
        angle = alpha * angle + (1 - alpha) * self.last_angle
        
        self.last_speed = speed
        self.last_angle = angle
        
        return speed, angle
    
    def step(self) -> Tuple[float, float]:
        """
        执行一步控制
        
        :return: (speed, angle) 速度和角度控制量
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        return self.calculate_control()
    
    def send_control(self, ip: str, port: int):
        """
        发送控制指令到车辆
        
        :param ip: 目标IP地址
        :param port: 目标端口
        """
        speed, angle = self.step()
        send_ctrl(speed, angle, ip, port, self.car_communication, self.bias)
        
    def reset(self):
        """
        重置规划器状态
        """
        self.current_target_idx = 0
        self.is_finished = False
        self.stuck_counter = 0
        self.last_pos = np.array([0.0, 0.0])
        
    def get_progress(self) -> float:
        """
        获取轨迹跟踪进度
        
        :return: 进度百分比 (0-1)
        """
        if len(self.trajectory_points) == 0:
            return 1.0
        return min(self.current_target_idx / len(self.trajectory_points), 1.0)
    
    def get_remaining_distance(self) -> float:
        """
        获取剩余距离
        
        :return: 剩余距离
        """
        if self.is_finished:
            return 0.0
        
        total_distance = 0.0
        
        # 计算到当前目标点的距离
        if self.current_target_idx < len(self.trajectory_points):
            current_target = self.trajectory_points[self.current_target_idx]
            total_distance += np.linalg.norm(self.current_pos - current_target)
        
        # 计算剩余轨迹点之间的距离
        for i in range(self.current_target_idx, len(self.trajectory_points) - 1):
            point1 = self.trajectory_points[i]
            point2 = self.trajectory_points[i + 1]
            total_distance += np.linalg.norm(point2 - point1)
        
        return total_distance
    
    def create_log_file(self):
        """创建日志文件记录实际路径"""
        log_dir = "trajectory_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.name:
            self.log_file = open(f"{log_dir}/trajectory_{timestamp}_{self.name}.log", "w")
        else:
            self.log_file = open(f"{log_dir}/trajectory_{timestamp}.log", "w")

        self.log_file.write("time,x,y,heading,target_x,target_y,target_idx\n")
            
    def close_log_file(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# 辅助函数
def send_ctrl(speed, angle, ip, port, car_communication, bias=0):
    """
    发送控制指令，最多尝试三次
    
    :param speed: 速度
    :param angle: 角度
    :param ip: IP地址
    :param port: 端口
    :param car_communication: 通信socket
    """
    buffer = cvt_ctrl_to_car_ctrl(speed, angle, bias)
    command = "<%d,%d,%d,%d>" % (int(buffer[0]), int(buffer[1]), int(buffer[2]), int(buffer[3]))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            car_communication.sendto(command.encode(), (ip, port))
            return  # Success, exit the function
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt failed
                rospy.logerr(f"Failed to send control command after {max_retries} attempts: {str(e)}")
                raise  # Re-raise the exception if you want the caller to handle it
            rospy.logwarn(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(0.1)  # Small delay before retry

def cvt_ctrl_to_car_ctrl(speed, angle, bias=0):
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

# 车辆运动模拟
def simulate_vehicle_motion(current_pos, current_heading, speed, angle, dt=0.1):
    """
    模拟车辆运动 
    
    :param current_pos: 当前位置 [x, y]
    :param current_heading: 当前朝向（弧度）
    :param speed: 速度控制量
    :param angle: 角度控制量
    :param dt: 时间步长
    :return: 新位置, 新朝向
    """
    # 将控制量转换为实际的速度和角速度
    actual_speed = speed * 0.1  # 缩放因子，调整车辆移动速度
    actual_angular_velocity = angle * 0.02  # 缩放因子，调整车辆转向速度
    
    # 更新朝向
    new_heading = current_heading + actual_angular_velocity * dt
    
    # 更新位置
    new_x = current_pos[0] + actual_speed * math.cos(new_heading) * dt
    new_y = current_pos[1] + actual_speed * math.sin(new_heading) * dt
    
    return np.array([new_x, new_y]), new_heading

class StateReceiver:
    def __init__(self):
        self.current_pos = np.zeros(2)
        self.current_heading = 0.0
        self.current_twist = None
        self.lock = threading.Lock()

    def pose_callback(self, msg: PoseStamped):
        """
        回调函数：接收小车位置和朝向
        """
        with self.lock:
            self.current_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y
            ])
            # 四元数转偏航角
            q = msg.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_heading = math.atan2(siny_cosp, cosy_cosp)

    def twist_callback(self, msg: TwistStamped):
        """
        回调函数：接收小车速度信息
        """
        with self.lock:
            self.current_twist = msg

    def get_state(self):
        """
        返回当前小车状态：(位置, 朝向, 速度信息)
        """
        with self.lock:
            return self.current_pos.copy(), self.current_heading, self.current_twist

# 使用示例
if __name__ == "__main__":
    current_twist = None
    ip_7 = "192.168.1.207"
    port = int(12345)
    # 定义轨迹点
    trajectory = [(249.82962346682265, 403.38896104873334), (263.3746038268231, 388.25394111597245), (276.9195841868235, 373.1189211832118), (290.4645645468239, 357.983901250451), (304.0095449068243, 342.84888131769037), (317.5545252668247, 327.7138613849295), (317.0728697443277, 313.36780672364785), (316.5912142218307, 299.021752062366), (316.1095586993338, 284.67569740108434), (315.62790317683675, 270.32964273980247), (315.14624765433985, 255.98358807852082), (314.6645921318428, 241.63753341723896), (328.20957249184323, 226.5025134844783), (341.75455285184364, 211.36749355171742), (355.29953321184405, 196.23247361895676), (369.32616909434137, 195.44350834747775), (383.3528049768388, 194.65454307599884), (397.3794408593361, 193.86557780451983), (411.40607674183354, 193.07661253304082), (425.43271262433086, 192.2876472615618), (439.4593485068283, 191.4986819900828), (453.4859843893256, 190.70971671860377), (467.51262027182304, 189.92075144712476), (481.53925615432036, 189.13178617564586), (495.5658920368177, 188.34282090416684), (509.5925279193151, 187.55385563268783), (523.6191638018124, 186.7648903612088), (537.6457996843098, 185.9759250897298), (551.6724355668072, 185.18695981825078), (565.6990714493046, 184.39799454677177), (579.7257073318019, 183.60902927529287), (593.7523432142993, 182.82006400381385), (607.7789790967967, 182.03109873233484), (621.8056149792941, 181.24213346085583), (635.8322508617914, 180.4531681893768), (649.8588867442888, 179.6642029178978), (663.8855226267862, 178.87523764641878), (677.9121585092835, 178.08627237493977), (691.9387943917809, 177.29730710346087), (705.9654302742782, 176.50834183198185), (719.9920661567756, 175.71937656050284), (734.018702039273, 174.93041128902382), (748.0453379217704, 174.1414460175448), (762.0719738042677, 173.3524807460658), (776.0986096867651, 172.56351547458678), (790.1252455692625, 171.77455020310776), (804.1518814517599, 170.98558493162886), (818.6601728567542, 184.5426743214315), (832.6868087392517, 183.75370904995248), (846.713444621749, 182.96474377847346), (861.2217360267432, 196.52183316827632), (875.2483719092406, 195.7328678967973), (889.756663314235, 209.28995728659993), (903.7832991967324, 208.50099201512103), (918.2915906017267, 222.05808140492388), (932.3182264842239, 221.26911613344487), (946.8265178892184, 234.8262055232475), (960.8531537717158, 234.03724025176848), (975.3614451767103, 247.59432964157133), (989.3880810592075, 246.80536437009232), (1003.4147169417049, 246.01639909861342), (1017.9230083466994, 259.57348848841605), (1031.9496442291966, 258.78452321693703), (1046.4579356341908, 272.3416126067399), (1060.4845715166882, 271.5526473352609), (1074.9928629216827, 285.1097367250635), (1089.0194988041799, 284.3207714535845), (1103.5277902091743, 297.87786084338745), (1117.5544260916718, 297.08889557190844), (1131.5810619741692, 296.2999303004294), (1146.0893533791634, 309.85701969023205), (1154.759152784109, 568.0860035933036), (1146.0893533791634, 309.85701969023205), (1137.419553974218, 51.62803578716034), (1128.7497545692722, -206.60094811591125), (1212.4279143417598, -225.68079440606675), (1221.0977137467053, 32.54818949700484), (1229.7675131516507, 290.77717340007644), (1238.4373125565962, 549.0061573031479), (1323.0787833740774, 558.6184203355557), (1314.408983969132, 300.389436432484), (1305.7391845641864, 42.160452529412396), (1297.0693851592407, -216.0685313736592)]
    # 创建轨迹跟踪器 - 使用稳定参数
    planner = TrajectoryPlanner(
        trajectory_points=trajectory,
        lookahead_distance=300,
        max_speed=100.0,
        min_speed=10.0,
        goal_tolerance=80,
        max_angle_control=60.0
    )
    
    # 创建状态接收器
    receiver = StateReceiver()
    rospy.init_node("trajectory_follower", anonymous=True)
    rospy.Subscriber("/vrpn_client_node/vehicle_1/pose", PoseStamped, receiver.pose_callback, queue_size=1)
    rospy.Subscriber("/vrpn_client_node/vehicle_1/twist", TwistStamped, receiver.twist_callback, queue_size=1)

    rate = rospy.Rate(10)  # 控制频率 10Hz
    print("🚗 正在开始轨迹跟踪...")

    while not rospy.is_shutdown() and not planner.is_finished:
        pos, heading, twist = receiver.get_state()
        with receiver.lock:
            planner.update_position(pos[0], pos[1], heading)
            planner.current_twist = twist
        try:
            planner.send_control(ip_7, port)
        except Exception as e:
            rospy.logerr("Critical communication failure, stopping: " + str(e))
            break  # Or handle it differently if you prefer
        rate.sleep()

    if planner.is_finished:
        print("✅ 轨迹跟踪成功完成！")
        if planner.log_file:
            planner.close_log_file()
            
    else:
        car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send_ctrl(0, 0, ip_7, port, car_communication)
        print("❌ ROS 终止或控制中断")