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
    位置式PID实现
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
        self.value = self.kp * self.err + self.ki * \
            self.err_all + self.kd * (self.err - self.err_last)
        self.update()
        self.value = np.clip(self.value, self.lower, self.upper)
        return self.value

    def update(self):
        self.err_last = self.err
        self.err_all = self.err_all + self.err

    def reset(self):
        self.err = 0
        self.err_last = 0
        self.err_all = 0
        print("🐞 PID Reset!") # 增加一个打印，方便调试

    def set_target(self, target):
        self.target = target

class TrajectoryPlanner:
    """
    轨迹跟踪规划器
    """
    def __init__(self,
                 trajectory_points: List[Tuple[float, float]],
                 lookahead_distance: float = 80,
                 max_speed: float = 100.0,
                 min_speed: float = 20.0,
                 goal_tolerance: float = 30,
                 max_angle_control: float = 50.0,
                 turn_in_place_threshold: float = 10.0,
                 prediction_horizon: float = 0.2,
                 bias=0,
                 name=None):
        self.trajectory_points = self.interpolate_path(trajectory_points, spacing=20)
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.goal_tolerance = goal_tolerance
        self.max_angle_control = max_angle_control
        self.turn_in_place_threshold = turn_in_place_threshold
        self.prediction_horizon = prediction_horizon
        self.bias = bias
        self.name = name

        self.current_pos = np.array([0.0, 0.0])
        self.current_twist = None
        self.trajectory_history = []
        self.current_heading = 0.0
        self.current_target_idx = 0
        
        self.is_finished = False
        self.last_time = time.time()
        
        self.car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.log_file = None
        self.create_log_file()
        
        self.angle_pid = PID_posi(
            kp=1.2, ki=0.1, kd=0.8,
            target=0.0,
            upper=self.max_angle_control,
            lower=-self.max_angle_control
        )
        
        self.min_angle_control = 30.0
        self.last_speed = 0.0
        self.last_angle = 0.0

        # ==================== 新增：卡住检测相关参数 ====================
        self.stuck_velocity_threshold = 0.01  # (米/秒) 判断车辆静止的速度阈值
        self.stuck_duration_threshold = 3.0   # (秒) 持续静止多久后判断为卡住
        self.is_stuck = False                 # 是否处于卡住自救状态
        self.stuck_start_time = 0.0           # 开始自救的时间戳
        self.stuck_check_timer = time.time()  # 用于检测是否卡住的计时器
        # ============================================================

    def update_position(self, x: float, y: float, heading: float):
        self.current_pos = np.array([x, y])
        self.current_heading = heading
        self.trajectory_history.append((x, y))

        if self.log_file:
            target_x, target_y, target_idx = self.find_target_point(self.current_pos)
            self.log_file.write(
                f"{time.time()},{x},{y},{heading},{target_x},{target_y},{target_idx}\n"
            )
            self.log_file.flush()
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        return self.trajectory_history.copy()   
      
    def interpolate_path(self, points, spacing=30.0):
        interpolated = []
        for i in range(len(points) - 1):
            p0 = np.array(points[i])
            p1 = np.array(points[i + 1])
            segment_vec = p1 - p0
            segment_len = np.linalg.norm(segment_vec)
            if segment_len < 1e-6: continue
            direction = segment_vec / segment_len
            num_points = max(int(segment_len // spacing), 1)
            for j in range(num_points):
                new_point = p0 + direction * spacing * j
                interpolated.append(tuple(new_point))
        interpolated.append(tuple(points[-1]))
        return interpolated
    
    def find_target_point(self, from_pos: np.ndarray) -> Tuple[float, float, int]:
        best_idx = self.current_target_idx
        for i in range(self.current_target_idx, len(self.trajectory_points)):
            pt = self.trajectory_points[i]
            dist = np.linalg.norm(from_pos - pt)
            if dist >= self.lookahead_distance:
                best_idx = i
                break
        else:
            best_idx = len(self.trajectory_points) - 1
        self.current_target_idx = best_idx
        target_point = self.trajectory_points[best_idx]
        return target_point[0], target_point[1], best_idx

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def _predict_future_state(self, pos, heading, speed, angle, dt):
        actual_speed = speed * 0.1
        actual_angular_velocity = angle * 0.02
        new_heading = heading + actual_angular_velocity * dt
        new_pos = pos + np.array([
            actual_speed * math.cos(new_heading) * dt,
            actual_speed * math.sin(new_heading) * dt
        ])
        return new_pos, new_heading

    def calculate_control(self) -> Tuple[float, float]:
        """
        计算控制指令（速度和角速度）
        """
        # ==================== 新增：卡住检测与自救逻辑 ====================
        # 1. 检查是否正处于自救模式
        if self.is_stuck:
            # 检查自救旋转是否已满1秒
            if time.time() - self.stuck_start_time > 1.0:
                print("✅ 自救旋转完成，恢复正常控制。")
                self.is_stuck = False
                self.stuck_check_timer = time.time() # 重置检测计时器
                self.angle_pid.reset()               # 重置PID防止突变
            else:
                # 仍在1秒自救时间内，继续以最大角速度旋转
                print("🌪️ 正在执行自救：以最大角速度旋转...")
                return 0.0, self.max_angle_control # 返回0速度和最大角速度
        
        # 2. 检查是否到达终点（自救逻辑之后）
        if self.is_finished:
            return 0.0, 0.0
        
        # 3. 判断是否卡住
        # 条件：(1)收到twist数据 (2)上次指令速度>最小速度 (3)实际速度<阈值
        if self.current_twist and self.last_speed > self.min_speed:
            linear_velocity = self.current_twist.twist.linear.x
            if abs(linear_velocity) < self.stuck_velocity_threshold:
                # 速度很低，检查持续时间
                if time.time() - self.stuck_check_timer > self.stuck_duration_threshold:
                    print(f"🚨 检测到卡住！(持续 {self.stuck_duration_threshold}s 速度低于 {self.stuck_velocity_threshold} m/s)")
                    self.is_stuck = True
                    self.stuck_start_time = time.time()
                    return 0.0, self.max_angle_control # 立即开始自救旋转
            else:
                # 车辆在正常移动，重置计时器
                self.stuck_check_timer = time.time()
        else:
            # 没有发出前进指令或无twist数据，重置计时器
            self.stuck_check_timer = time.time()
        # =================================================================

        # --- 1. 预测辅助控制 ---
        predicted_pos, predicted_heading = self._predict_future_state(
            self.current_pos, self.current_heading, self.last_speed, self.last_angle, self.prediction_horizon
        )
        print(f"🔮 预测模块 | 当前位置: ({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}), "
              f"预测位置: ({predicted_pos[0]:.1f}, {predicted_pos[1]:.1f})")

        # --- 2. 寻找目标点 ---
        target_x, target_y, target_idx = self.find_target_point(from_pos=predicted_pos)
        
        # --- 3. 检查是否到达终点 ---
        distance_to_end = np.linalg.norm(self.current_pos - self.trajectory_points[-1])
        if target_idx >= len(self.trajectory_points) - 1 and distance_to_end < self.goal_tolerance:
            self.is_finished = True
            print("🎉 轨迹跟踪完成！")
            return 0.0, 0.0
        
        # --- 4. 计算误差 ---
        target_vector = np.array([target_x, target_y]) - predicted_pos
        target_distance = np.linalg.norm(target_vector)
        target_angle = math.atan2(target_vector[1], target_vector[0])
        angle_error = -self.normalize_angle(target_angle - predicted_heading)
        angle_error_deg = math.degrees(angle_error)

        # --- 5. 核心控制逻辑：【新】严格的“先转向，再直行” ---
        speed = 0.0
        angle = 0.0

        if abs(angle_error_deg) > self.turn_in_place_threshold:
            # 角度误差过大，执行原地转向
            print("🔄 模式: 原地转向 (速度=0)")
            speed = 0.0 # 速度严格为0
            angle = self.angle_pid.cal_output(angle_error) # 仅用PID计算角速度
        else:
            # 角度误差在容忍范围内，执行纯直行
            print("➡️  模式: 纯直行 (角速度=0)")
            # a. 速度控制：计算前进速度
            speed = self._calculate_adaptive_speed(target_distance, angle_error)
            # b. 角速度控制：严格设置为0
            angle = 0.0
            # c. 重置PID：为下一次可能的转向做准备，防止积分项累积
            self.angle_pid.reset()

        # --- 6. 应用电机物理限制 ---
        if speed > 0.1:
            speed = np.clip(speed, self.min_speed, self.max_speed)
        if 0 < abs(angle) < self.min_angle_control:
            angle = self.min_angle_control * np.sign(angle)
        
        speed = np.clip(speed, 0, self.max_speed)
        angle = np.clip(angle, -self.max_angle_control, self.max_angle_control)
        
        self.last_speed = speed
        self.last_angle = angle
        
        # --- 7. 打印调试信息 ---
        self._print_debug_info(target_x, target_y, target_distance, angle_error_deg, speed, angle)
        
        return speed, angle

    def _calculate_adaptive_speed(self, distance: float, angle_error: float) -> float:
        if distance > 500: base_speed = self.max_speed
        elif distance > 300: base_speed = self.max_speed * 0.8
        else: base_speed = self.max_speed * 0.6
        angle_factor = 1.0 - min(abs(angle_error) / (math.pi / 2), 0.8)
        return base_speed * angle_factor

    def _print_debug_info(self, target_x, target_y, target_distance, angle_error_deg, speed, angle):
        print(f"📊 {self.name} 状态报告:")
        print(f"   当前位置: ({self.current_pos[0]:.2f}, {self.current_pos[1]:.2f})")
        print(f"   当前朝向: {self.current_heading * 180 / math.pi:.1f}°")
        print(f"   目标点: ({target_x:.1f}, {target_y:.1f})")
        print(f"   与目标距离: {target_distance:.2f}mm")
        print(f"   角度误差: {angle_error_deg:.1f}°")
        print(f"   PID输出(角): {self.angle_pid.value:.1f}")
        print(f"   最终控制: 速度={speed:.1f}, 角速度={angle:.1f}\n" + "-"*30)
    
    def step(self) -> Tuple[float, float]:
        self.last_time = time.time()
        return self.calculate_control()
    
    def send_control(self, ip: str, port: int):
        speed, angle = self.step()
        send_ctrl(speed, angle, ip, port, self.car_communication, self.bias)
        
    def reset(self):
        self.current_target_idx = 0
        self.is_finished = False
        
    def create_log_file(self):
        log_dir = "trajectory_logs"
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_dir}/trajectory_{timestamp}_{self.name}.log" if self.name else f"{log_dir}/trajectory_{timestamp}.log"
        self.log_file = open(log_filename, "w")
        self.log_file.write("time,x,y,heading,target_x,target_y,target_idx\n")
            
    def close_log_file(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# ... (辅助函数 和 StateReceiver 类保持不变) ...
def send_ctrl(speed, angle, ip, port, car_communication, bias=0):
    buffer = cvt_ctrl_to_car_ctrl(speed, angle, bias)
    command = "<%d,%d,%d,%d>" % (int(buffer[0]), int(buffer[1]), int(buffer[2]), int(buffer[3]))
    max_retries = 3
    for attempt in range(max_retries):
        try:
            car_communication.sendto(command.encode(), (ip, port))
            return
        except Exception as e:
            if attempt == max_retries - 1:
                rospy.logerr(f"Failed to send control command after {max_retries} attempts: {str(e)}")
                raise
            rospy.logwarn(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(0.1)

def cvt_ctrl_to_car_ctrl(speed, angle, bias=0):
    buffer = np.zeros(4)
    left_speed = speed - angle - bias
    right_speed = speed + angle + bias
    buffer[0] = max(-100, min(100, left_speed))
    buffer[1] = max(-100, min(100, right_speed))
    buffer[2] = max(-100, min(100, left_speed))
    buffer[3] = max(-100, min(100, right_speed))
    return buffer

class StateReceiver:
    def __init__(self):
        self.current_pos = np.zeros(2)
        self.current_heading = 0.0
        self.current_twist = None
        self.lock = threading.Lock()

    def pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.current_pos = np.array([msg.pose.position.x, msg.pose.position.y])
            q = msg.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_heading = math.atan2(siny_cosp, cosy_cosp)

    def twist_callback(self, msg: TwistStamped):
        with self.lock:
            self.current_twist = msg

    def get_state(self):
        with self.lock:
            return self.current_pos.copy(), self.current_heading, self.current_twist
            
if __name__ == "__main__":
    ip_7 = "192.168.1.208"
    port = int(12345)
    
    trajectory = [(100,100),(1100,100),(1100,1100),(100,1100),(100,100)]
    
    planner = TrajectoryPlanner(
        trajectory_points=trajectory,
        lookahead_distance=60,
        max_speed=80.0,
        min_speed=20.0,
        goal_tolerance=30,
        max_angle_control=60.0,
        turn_in_place_threshold=15, # 建议将阈值调小，如5度，以便更精确地对准方向后再直行
        prediction_horizon=0.15,
        bias = 0
    )
    
    receiver = StateReceiver()
    rospy.init_node("trajectory_follower", anonymous=True)
    rospy.Subscriber("/vrpn_client_node/vehicle_1/pose", PoseStamped, receiver.pose_callback, queue_size=1)
    rospy.Subscriber("/vrpn_client_node/vehicle_1/twist", TwistStamped, receiver.twist_callback, queue_size=1)

    rate = rospy.Rate(10)
    print("🚗 正在开始轨迹跟踪 (模式: Turn-then-Drive)...")
    print(f"控制器参数: lookahead={planner.lookahead_distance}, max_speed={planner.max_speed}, min_speed={planner.min_speed}, "
          f"turn_threshold={planner.turn_in_place_threshold}°, prediction={planner.prediction_horizon}s")

    while not rospy.is_shutdown() and not planner.is_finished:
        pos, heading, twist = receiver.get_state()
        if np.all(pos == 0):
            print("⏳ 等待有效的初始位姿数据...")
            rate.sleep()
            continue
            
        with receiver.lock:
            planner.update_position(pos[0], pos[1], heading)
            planner.current_twist = twist
        try:
            planner.send_control(ip_7, port)
        except Exception as e:
            rospy.logerr("Critical communication failure, stopping: " + str(e))
            break
        rate.sleep()

    if planner.is_finished:
        print("✅ 轨迹跟踪成功完成！")
        send_ctrl(0, 0, ip_7, port, planner.car_communication)
        planner.close_log_file()
    else:
        car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send_ctrl(0, 0, ip_7, port, car_communication)
        print("❌ ROS 终止或控制中断，已发送停止指令。")