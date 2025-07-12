from camera import HikvisionCamera as Camera
from coordinate_mapper import CoordinateMapper
import test
import algorithms
import os
import time
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from typing import Tuple, Dict, List, Optional, Callable
import predict
import threading
from enum import Enum
import json
from controller import TrajectoryPlanner, StateReceiver, send_ctrl

class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    MAPPING = "mapping"
    READY = "ready"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"


class VehicleControlSystem:
    """多车控制系统主类"""
    
    def __init__(self, camera_index: int = 0, capture_dir: str = "./captures"):
        """
        初始化系统
        
        Args:
            camera_index: 相机设备索引
            capture_dir: 图像捕获目录
        """
        self.camera_index = camera_index
        self.capture_dir = capture_dir
        self.state = SystemState.IDLE
        
        # 核心组件
        self.camera = None
        self.mapper = None
        
        # 数据存储
        self.vehicles = []  # 小车信息
        self.obstacles = []  # 障碍物信息
        self.destinations = []  # 目的地信息
        self.grid_vehicles = []
        self.grid_obstacles = []
        self.grid_destinations = []
        self.car_id_mapping = {}  # 小车ID映射
        self.path_results = {}  # 路径规划结果
        
        # ROS配置
        self.car_topics = [
            "/vrpn_client_node/vehicle_1/pose",
            "/vrpn_client_node/vehicle_2/pose", 
            "/vrpn_client_node/vehicle_3/pose"
        ]
        
        # 控制参数
        self.detection_interval = 5.0  # 检测间隔（秒）
        self.camera_rotation = -19  # 相机旋转角度
        self.mapper_file = "coordinate_mapper.pkl"
        
        # 回调函数
        self.status_callback = None  # 状态更新回调
        self.error_callback = None  # 错误回调
        self.progress_callback = None  # 进度回调
        
        # 线程控制
        self.running = False
        self.main_thread = None
        self._stop_event = threading.Event()
        
        # 控制模块相关
        self.trajectory_planners = {}  # 存储每个小车的轨迹规划器
        self.state_receivers = {}      # 存储每个小车的状态接收器
        self.car_ips = {               # 小车IP配置
            0: "192.168.1.208",
            1: "192.168.1.207",  # 根据实际情况配置
            2: "192.168.1.205"
        }
        self.car_bias = {
            0: -5,
            1: 7,
            2: 0
        }
        self.car_port = 12345
        
    def set_callbacks(self, status_callback: Callable = None, 
                     error_callback: Callable = None,
                     progress_callback: Callable = None):
        """设置回调函数"""
        self.status_callback = status_callback
        self.error_callback = error_callback
        self.progress_callback = progress_callback
        
    def set_control_module(self, control_module):
        """设置控制模块"""
        self.control_module = control_module

    def set_car_bias(self, car_id: int, bias: int):
        """设置指定小车的bias"""
        self.car_bias[car_id] = bias
        print(f"小车{car_id}的bias设置为: {bias}")

    def set_all_car_bias(self, bias_dict: Dict[int, int]):
        """批量设置所有小车的bias"""
        self.car_bias.update(bias_dict)
        print(f"小车bias配置: {self.car_bias}")

    def _update_state(self, new_state: SystemState, message: str = ""):
        """更新系统状态"""
        self.state = new_state
        if self.status_callback:
            self.status_callback(new_state, message)
        print(f"状态更新: {new_state.value} - {message}")
        
    def _report_error(self, error_msg: str):
        """报告错误"""
        self._update_state(SystemState.ERROR, error_msg)
        if self.error_callback:
            self.error_callback(error_msg)
        print(f"错误: {error_msg}")
        
    def _report_progress(self, progress: float, message: str = ""):
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(progress, message)
        print(f"进度: {progress:.1f}% - {message}")
        
    def initialize_system(self) -> bool:
        """初始化系统"""
        try:
            self._update_state(SystemState.INITIALIZING, "正在初始化系统...")
            
            # 创建目录
            if not os.path.exists(self.capture_dir):
                os.makedirs(self.capture_dir)
                
            # 初始化相机
            self.camera = Camera(device_index=self.camera_index)
            devices = self.camera.list_devices()
            print(f"可用设备: {devices}")
            
            if not self.camera.connect():
                self._report_error("相机连接失败")
                return False
                
            print("相机连接成功")
            
            # 初始化坐标映射器
            self.mapper = CoordinateMapper()
                
            self._update_state(SystemState.READY, "系统初始化完成")
            return True
            
        except Exception as e:
            self._report_error(f"系统初始化失败: {str(e)}")
            return False
            
    def setup_coordinate_mapping(self) -> bool:
        """设置坐标映射"""
        try:
            self._update_state(SystemState.MAPPING, "正在设置坐标映射...")
            
            # 尝试加载现有映射器
            loaded_mapper = CoordinateMapper.load_mapper(self.mapper_file)
            if loaded_mapper:
                self.mapper = loaded_mapper
                print("已加载坐标映射器")
                return True
                
            print("未找到坐标映射器文件，正在初始化...")
            
            # 拍摄初始化图像
            init_image_path = f"{self.capture_dir}/init_mapping.jpg"
            self.camera.capture_rotated_image(file_path=init_image_path, angle=self.camera_rotation)
            
            # 检测小车获取图像坐标
            vehicle_img_coords = self.mapper.detect_vehicle(
                path=init_image_path, 
                model_path="best.pt", 
                show_results=False
            )
            
            if vehicle_img_coords is None:
                self._report_error("未检测到小车目标，请检查图像或模型")
                return False
                
            print(f"图像坐标: {vehicle_img_coords}")
            
            # 获取ROS实际坐标
            print("等待ROS小车位置...")
            vehicle_real_coords = []
            
            for topic in self.car_topics:
                real_pos = self._get_car_position(topic)
                if real_pos is None:
                    self._report_error(f"无法获取{topic}的位置信息")
                    return False
                print(f"{topic} -> {real_pos}")
                vehicle_real_coords.append(real_pos)
                
            print(f"实际坐标: {vehicle_real_coords}")
            
            # 初始化坐标映射器
            if len(vehicle_img_coords) == len(vehicle_real_coords):
                self.mapper.initialize_transform(vehicle_img_coords, vehicle_real_coords)
                self.mapper.save_mapper(self.mapper_file)
                print("坐标映射初始化完成")
                return True
            else:
                self._report_error("图像与实际坐标数量不一致")
                return False
                
        except Exception as e:
            self._report_error(f"坐标映射设置失败: {str(e)}")
            return False
            
    def perform_initial_detection(self) -> bool:
        """执行初始检测，确定小车、障碍物、目的地"""
        try:
            self._update_state(SystemState.PLANNING, "正在执行初始检测...")
            
            # 拍摄检测图像
            timestamp = int(time.time())
            image_path = f"{self.capture_dir}/initial_detection_{timestamp}.jpg"
            self.camera.capture_rotated_image(file_path=image_path, angle=self.camera_rotation)
            
            # YOLO检测
            results = predict.detect_objects(path=image_path, show_results=False)
            
            # 保存检测结果
            predict.save_detection_results(results, save_dir="detection_results")
            
            # 解析检测结果
            if not results or not results[0]:
                self._report_error("初始检测失败")
                return False
                
            detection_data = results[0]
            # 提取小车信息
            vehicle_img_coords = detection_data.get('all_vehicles', [])
            if not vehicle_img_coords:
                self._report_error("未检测到小车")
                return False
                
            for i, img_coord in enumerate(vehicle_img_coords):
                vehicle_img_coords[i] = self.get_center_from_bbox(img_coord)

            # 获取ROS实际坐标
            vehicle_real_coords = []
            for topic in self.car_topics:
                real_pos = self._get_car_position(topic)
                if real_pos is None:
                    self._report_error(f"无法获取{topic}的位置信息")
                    return False
                vehicle_real_coords.append(real_pos)
                
            # 建立ID映射关系
            self._establish_vehicle_id_mapping(vehicle_img_coords, vehicle_real_coords)
            
            # 存储小车信息
            self.vehicles = []
            for i, img_coord in enumerate(vehicle_img_coords):
                real_coord = self.mapper.map_to_real_coords(img_coord)
                self.vehicles.append({
                    'id': i,
                    'img_coord': img_coord,
                    'real_coord': real_coord,
                    'ros_coord': vehicle_real_coords[i] if i < len(vehicle_real_coords) else None
                })
                
            # 提取障碍物信息
            obstacle_img_coords = detection_data.get('obstacle', [])
            self.obstacles = []
            for i, img_coord in enumerate(obstacle_img_coords):
                obstacle_img_coords[i] = self.get_center_from_bbox(img_coord)

            for i, img_coord in enumerate(obstacle_img_coords):
                real_coord = self.mapper.map_to_real_coords(img_coord)
                self.obstacles.append({
                    'id': i,
                    'img_coord': img_coord,
                    'real_coord': real_coord
                })
                
            # 提取目的地信息
            destination_img_coords = detection_data.get('destination', [])
            self.destinations = []
            for i, img_coord in enumerate(destination_img_coords):
                destination_img_coords[i] = self.get_center_from_bbox(img_coord)

            for i, img_coord in enumerate(destination_img_coords):
                real_coord = self.mapper.map_to_real_coords(img_coord)
                self.destinations.append({
                    'id': i,
                    'img_coord': img_coord,
                    'real_coord': real_coord
                })
                
            print(f"检测完成 - 小车: {len(self.vehicles)}, 障碍物: {len(self.obstacles)}, 目的地: {len(self.destinations)}")
            return True
            
        except Exception as e:
            self._report_error(f"初始检测失败: {str(e)}")
            return False
        
    def load_json_file(self, file_path=None):
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if self.validate_json_data(data):
                self.grid_vehicles = data.get("all_vehicles", [])
                self.grid_obstacles = data.get("obstacle", [])
                self.grid_destinations = data.get("destination", [])
                
                if hasattr(self, 'file_status'):
                    self.file_status.config(text=f"文件导入成功: {os.path.basename(file_path)}", foreground="green")
                self.update_data_display()
                self.update_visualization()
            else:
                raise ValueError("JSON格式不正确")
                
        except Exception as e:
            if hasattr(self, 'file_status'):
                self.file_status.config(text="文件导入失败", foreground="red")
    
    def validate_json_data(self, data):
        """验证JSON数据格式"""
        required_keys = ["all_vehicles", "obstacle", "destination"]
        return all(key in data for key in required_keys)
            
    def plan_paths(self, command: str = "") -> bool:
        """路径规划"""
        try:
            self._update_state(SystemState.PLANNING, "正在进行路径规划...")
            
            self.load_json_file(file_path="detection_results/detection_results.json")

            # 创建路径规划对象
            obj = algorithms.PathPlanner(self.grid_vehicles, self.grid_obstacles, self.grid_destinations)

            # 调用LLM (可选)
            function_list = test.call_LLM(self.grid_vehicles, self.grid_destinations, command)
            
            if function_list is None:
                self._report_error("LLM调用失败")
                return False
            else:
                # 解释LLM结果并生成路径
                self.path_results = test.Interpret_function_list(function_list, obj)

            if not self.path_results:
                self._report_error("路径规划失败")
                return False
            
            for i in range(len(self.path_results)):
                self._create_trajectory_planners(task_index=i)
                # 等待任务完成 ...

            print(f"路径规划完成，共{len(self.path_results)}条路径")
            return True
            
        except Exception as e:
            self._report_error(f"路径规划失败: {str(e)}")
            return False
            
    def _create_trajectory_planners(self, task_index=0):
        """
        为当前任务（第 task_index 个）创建轨迹规划器
        """
        if not self.path_results:
            self._report_error("路径结果为空")
            return

        if task_index >= len(self.path_results):
            self._report_error(f"任务索引 {task_index} 超出范围")
            return

        results_dict = self.path_results[task_index]  # 取出当前任务路径字典

        for vehicle_id, path in results_dict.items():
            if path:
                car_bias = self.car_bias.get(vehicle_id, 0)
                planner = TrajectoryPlanner(
                    trajectory_points=path,
                    lookahead_distance=300,
                    max_speed=100.0,
                    min_speed=10.0,
                    goal_tolerance=80,
                    max_angle_control=60.0,
                    bias=car_bias
                )
                self.trajectory_planners[vehicle_id] = planner

                # 创建状态接收器
                receiver = StateReceiver()
                self.state_receivers[vehicle_id] = receiver

                # 订阅 ROS 话题
                topic_index = self.car_id_mapping.get(vehicle_id, vehicle_id)
                if topic_index < len(self.car_topics):
                    pose_topic = self.car_topics[topic_index]
                    twist_topic = pose_topic.replace('/pose', '/twist')

                    rospy.Subscriber(pose_topic, PoseStamped, receiver.pose_callback, queue_size=1)
                    rospy.Subscriber(twist_topic, TwistStamped, receiver.twist_callback, queue_size=1)

    
    def execute_mission(self) -> bool:
        """执行任务"""
        try:
            self._update_state(SystemState.EXECUTING, "正在执行任务...")
            
            if not self.path_results:
                self._report_error("没有可执行的路径")
                return False
            
            rate = rospy.Rate(10)  # 10Hz控制频率
                
            # 任务执行主循环
            while not self._stop_event.is_set() and not rospy.is_shutdown():
                # 更新小车位置
                if not self._update_vehicle_positions():
                    break
                    
                self._execute_trajectory_control()

                # 检查任务完成状态
                if self._check_mission_completion():
                    self._update_state(SystemState.COMPLETED, "任务完成")
                    break
                    
                rate.sleep()
                
            return True
            
        except Exception as e:
            self._report_error(f"任务执行失败: {str(e)}")
            return False
        
    def _execute_trajectory_control(self):
        """执行轨迹跟踪控制"""
        for vehicle_id, planner in self.trajectory_planners.items():
            if vehicle_id in self.state_receivers and not planner.is_finished:
                receiver = self.state_receivers[vehicle_id]
                pos, heading, twist = receiver.get_state()
                
                with receiver.lock:
                    planner.update_position(pos[0], pos[1], heading)
                    planner.current_twist = twist
                
                try:
                    car_ip = self.car_ips.get(vehicle_id, "192.168.1.207")
                    planner.send_control(car_ip, self.car_port)
                except Exception as e:
                    print(f"小车{vehicle_id}控制发送失败: {str(e)}") 

    def start_mission(self, command: str = "") -> bool:
        """启动任务"""
        if self.running:
            self._report_error("任务已在运行中")
            return False
        
        try:
            self.running = True
            self._stop_event.clear()
            
            # 在主线程中初始化ROS节点
            if not rospy.get_node_uri():
                rospy.init_node('vehicle_control_system', anonymous=True)
                print("ROS节点初始化完成")
            
            # 1. 初始化系统（不包括ROS）
            if not self.initialize_system():
                return False
                
            # 2. 设置坐标映射
            if not self.setup_coordinate_mapping():
                return False
                
            # 3. 执行初始检测
            if not self.perform_initial_detection():
                return False
                
            # 4. 路径规划
            if not self.plan_paths(command):
                return False
                
            # 5. 执行任务
            return self.execute_mission()
            
        except Exception as e:
            self._report_error(f"任务执行异常: {str(e)}")
            return False
        finally:
            self.running = False

    def stop_mission(self):
        """停止任务"""
        self._stop_event.set()
        self.running = False
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5)
            
    def pause_mission(self):
        """暂停任务"""
        self._update_state(SystemState.PAUSED, "任务已暂停")
        
    def resume_mission(self):
        """恢复任务"""
        self._update_state(SystemState.EXECUTING, "任务已恢复")
        
    def cleanup(self):
        """清理资源"""
        # 停止所有小车
        for vehicle_id in self.car_ips:
            try:
                car_ip = self.car_ips[vehicle_id]
                # 发送停止命令
                import socket
                car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                send_ctrl(0, 0, car_ip, self.car_port, car_communication)
            except Exception as e:
                print(f"停止小车{vehicle_id}失败: {str(e)}")
        
        # 关闭日志文件
        for planner in self.trajectory_planners.values():
            if hasattr(planner, 'log_file') and planner.log_file:
                planner.close_log_file()
            self.stop_mission()
        if self.camera:
            self.camera.disconnect()
            print("相机已断开连接")
            
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'state': self.state.value,
            'running': self.running,
            'vehicles': len(self.vehicles),
            'obstacles': len(self.obstacles),
            'destinations': len(self.destinations),
            'has_paths': bool(self.path_results)
        }
        
    def _get_car_position(self, topic: str, timeout: float = 5.0) -> Optional[Tuple[float, float]]:
        """获取小车位置"""
        try:
            msg = rospy.wait_for_message(topic, PoseStamped, timeout=timeout)
            pos = msg.pose.position
            return (pos.x, pos.y)
        except Exception as e:
            print(f"获取{topic}位置失败: {str(e)}")
            return None
            
    def _establish_vehicle_id_mapping(self, vehicle_img_coords: List, vehicle_real_coords: List):
        """建立小车ID映射关系"""
        self.car_id_mapping = {}
        
        for i, img_coord in enumerate(vehicle_img_coords):
            # 将图像坐标映射到实际坐标
            mapped_coord = self.mapper.map_to_real_coords(img_coord)
            
            # 找到最接近的ROS小车
            closest_idx = None
            min_dist = float('inf')
            for j, real_coord in enumerate(vehicle_real_coords):
                dist = ((mapped_coord[0] - real_coord[0])**2 + 
                       (mapped_coord[1] - real_coord[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j
                    
            if closest_idx is not None:
                self.car_id_mapping[i] = closest_idx
                
        print(f"小车ID映射: {self.car_id_mapping}")
        
    def _update_vehicle_positions(self) -> bool:
        """更新小车位置"""
        try:
            # 拍摄当前图像
            timestamp = int(time.time())
            image_path = f"{self.capture_dir}/capture_{timestamp}.jpg"
            self.camera.capture_rotated_image(file_path=image_path, angle=self.camera_rotation)
            
            # YOLO检测
            results = predict.detect_objects(path=image_path, show_results=False)
            # 保存检测结果
            predict.save_detection_results(results, save_dir="detection_results")
            
            if not results or not results[0]:
                print("未检测到小车，等待下一次检测...")
                return True
                
            vehicle_img_coords = results[0].get('all_vehicles', [])
            if not vehicle_img_coords:
                print("未检测到小车，等待下一次检测...")
                return True
                
            for i, img_coord in enumerate(vehicle_img_coords):
                vehicle_img_coords[i] = self.get_center_from_bbox(img_coord)

            # 获取ROS实际坐标
            vehicle_real_coords = []
            for topic in self.car_topics:
                real_pos = self._get_car_position(topic)
                if real_pos is None:
                    return False
                vehicle_real_coords.append(real_pos)
                
            # 更新小车信息
            for i, vehicle in enumerate(self.vehicles):
                if i < len(vehicle_img_coords):
                    vehicle['img_coord'] = vehicle_img_coords[i]
                    vehicle['real_coord'] = self.mapper.map_to_real_coords(vehicle_img_coords[i])
                if i < len(vehicle_real_coords):
                    vehicle['ros_coord'] = vehicle_real_coords[i]
                    
            # 计算和打印误差
            for i, vehicle in enumerate(self.vehicles):
                if 'real_coord' in vehicle and 'ros_coord' in vehicle:
                    real_coord = vehicle['real_coord']
                    ros_coord = vehicle['ros_coord']
                    error = ((real_coord[0] - ros_coord[0])**2 + 
                            (real_coord[1] - ros_coord[1])**2)**0.5
                    print(f"小车{i+1}: 映射坐标={real_coord}, ROS坐标={ros_coord}, 误差={error:.2f}mm")
                    
            return True
            
        except Exception as e:
            print(f"更新小车位置失败: {str(e)}")
            return False
            
    def _check_mission_completion(self) -> bool:
        """检查任务完成状态"""
        # 检查所有轨迹规划器是否完成
        for planner in self.trajectory_planners.values():
            if not planner.is_finished:
                return False
        return True
    
    def get_center_from_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
# 使用示例
if __name__ == "__main__":
    # 创建系统实例
    system = VehicleControlSystem()
    
    # 设置回调函数
    def status_callback(state, message):
        print(f"状态回调: {state} - {message}")
        
    def error_callback(error_msg):
        print(f"错误回调: {error_msg}")
        
    def progress_callback(progress, message):
        print(f"进度回调: {progress}% - {message}")
        
    system.set_callbacks(status_callback, error_callback, progress_callback)
    
    try:
        # 启动任务
        if system.start_mission("小车0去往目的地0"):
            print("任务启动成功")
            
            # 等待任务完成或用户中断
            while system.running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("用户中断")
        
    finally:
        system.cleanup()