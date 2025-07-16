from camera import HikvisionCamera as Camera
from coordinate_mapper import CoordinateMapper
import test
import algorithms
import os
import time
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from typing import Tuple, Dict, List, Optional, Callable
import predict
import threading
from enum import Enum
import json
from controller import TrajectoryPlanner, StateReceiver, send_ctrl
from predict import batch_convert_to_image_coordinates
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
    
    def __init__(self, 
                 camera_index: int = 0, 
                 capture_dir: str = "./captures",
                 vehicle_ids: Optional[List[int]] = None,
                 car_ips: Optional[Dict[int, str]] = None,
                 car_bias: Optional[Dict[int, float]] = None,
                 car_port: int = 12345,
                 camera_rotation: int = -29):
        """
        初始化系统
        
        Args:
            camera_index: 相机设备索引
            capture_dir: 图像捕获目录
            vehicle_ids: 车辆ID列表，如[1, 2, 3]，用于生成对应的topics
            car_ips: 小车IP配置字典，格式为{车辆ID: IP地址}
            car_bias: 小车偏置配置字典，格式为{车辆ID: 偏置值}
            car_port: 小车通信端口
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
        self.grid_path_results = []
        self.path_results = []  # 路径规划结果
        
        # 设置小车配置
        self._setup_vehicle_config(vehicle_ids, car_ips, car_bias, car_port)
        
        # 控制参数
        self.camera_rotation = camera_rotation  # 相机旋转角度
        self.mapper_file = "coordinate_mapper.pkl"
        
        # 回调函数
        self.status_callback = None  # 状态更新回调
        self.error_callback = None  # 错误回调
        self.progress_callback = None  # 进度回调
        
        # 线程控制
        self.running = False
        self.main_thread = None
        self._stop_event = threading.Event()
        
        # 添加检测线程控制
        self.detection_thread = None
        self.detection_running = False
        
        # 控制模块相关
        self.trajectory_planners = {}  # 存储每个小车的轨迹规划器
        self.state_receivers = {}      # 存储每个小车的状态接收器
    
    def _setup_vehicle_config(self, 
                            vehicle_ids: Optional[List[int]] = None,
                            car_ips: Optional[Dict[int, str]] = None,
                            car_bias: Optional[Dict[int, float]] = None,
                            car_port: int = 12345,
                            camera_rotation: int = -29):
        """
        设置车辆配置
        
        Args:
            vehicle_ids: 车辆ID列表
            car_ips: 小车IP配置
            car_bias: 小车偏置配置
            car_port: 通信端口
        """
        # 设置默认车辆ID
        if vehicle_ids is None:
            vehicle_ids = [0,1,2]
        
        # 生成ROS话题
        self.car_topics = []
        for vehicle_id in vehicle_ids:
            topic = f"/vrpn_client_node/vehicle_{vehicle_id}/pose"
            self.car_topics.append(topic)
        
        # 设置小车IP配置
        if car_ips is None:
            # 默认IP配置（根据vehicle_ids的索引）
            default_ips = ["192.168.1.208", "192.168.1.205", "192.168.1.207"]
            self.car_ips = {}
            for i, vehicle_id in enumerate(vehicle_ids):
                if i < len(default_ips):
                    self.car_ips[i] = default_ips[i]  # 使用索引作为键
                else:
                    self.car_ips[i] = f"192.168.1.{200 + i}"  # 自动生成IP
        else:
            # 将vehicle_id映射到索引
            self.car_ips = {}
            for i, vehicle_id in enumerate(vehicle_ids):
                if vehicle_id in car_ips:
                    self.car_ips[i] = car_ips[vehicle_id]
                else:
                    self.car_ips[i] = f"192.168.1.{200 + i}"
        
        # 设置小车偏置配置
        if car_bias is None:
            # 默认偏置配置
            default_bias = [0, 0, 0]
            self.car_bias = {}
            for i, vehicle_id in enumerate(vehicle_ids):
                if i < len(default_bias):
                    self.car_bias[i] = default_bias[i]
                else:
                    self.car_bias[i] = 0  # 默认偏置为0
        else:
            # 将vehicle_id映射到索引
            self.car_bias = {}
            for i, vehicle_id in enumerate(vehicle_ids):
                if vehicle_id in car_bias:
                    self.car_bias[i] = car_bias[vehicle_id]
                else:
                    self.car_bias[i] = 0
        
        # 设置端口
        self.car_port = car_port
        self.camera_rotation = camera_rotation
        # 保存vehicle_ids用于其他功能
        self.vehicle_ids = vehicle_ids
    
    def get_vehicle_config(self) -> Dict:
        """
        获取当前车辆配置信息
        
        Returns:
            包含车辆配置信息的字典
        """
        return {
            "vehicle_ids": self.vehicle_ids,
            "car_topics": self.car_topics,
            "car_ips": self.car_ips,
            "car_bias": self.car_bias,
            "car_port": self.car_port
        }
    
    def update_vehicle_config(self, 
                            vehicle_ids: Optional[List[int]] = None,
                            car_ips: Optional[Dict[int, str]] = None,
                            car_bias: Optional[Dict[int, float]] = None,
                            car_port: Optional[int] = None):
        """
        更新车辆配置（仅在系统停止时允许）
        
        Args:
            vehicle_ids: 新的车辆ID列表
            car_ips: 新的小车IP配置
            car_bias: 新的小车偏置配置
            car_port: 新的通信端口
        
        Raises:
            RuntimeError: 如果系统正在运行
        """
        if self.state == SystemState.RUNNING:
            raise RuntimeError("Cannot update vehicle config while system is running")
        
        if vehicle_ids is not None:
            if car_port is not None:
                port = car_port
            else:
                port = self.car_port
            
            self._setup_vehicle_config(vehicle_ids, car_ips, car_bias, port)
        elif car_port is not None:
            self.car_port = car_port
            
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
    
    def optimize_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:   
        """
        合并路径中方向不变的连续点，仅保留拐点和端点
        适用于上下左右移动的路径点
        """
        if len(path) <= 2:
            return path.copy()

        optimized = [path[0]]  # 起点一定保留

        prev_dx = path[1][0] - path[0][0]
        prev_dy = path[1][1] - path[0][1]

        for i in range(1, len(path) - 1):
            curr_dx = path[i + 1][0] - path[i][0]
            curr_dy = path[i + 1][1] - path[i][1]

            # 如果方向变化了，保留这个拐点
            if (curr_dx, curr_dy) != (prev_dx, prev_dy):
                optimized.append(path[i])
            
            prev_dx = curr_dx
            prev_dy = curr_dy

        optimized.append(path[-1])  # 终点一定保留

        return optimized    
         
    def _detection_thread_worker(self):
        """检测线程工作函数"""
        print("检测线程启动")
        
        while self.detection_running and not self._stop_event.is_set():
            try:
                # 执行检测和更新
                self._update_vehicle_positions()
                
                # 等待1秒
                if not self._stop_event.wait(1.0):  # 使用Event的wait方法，支持中断
                    continue
                else:
                    break  # 收到停止信号
                    
            except Exception as e:
                print(f"检测线程异常: {str(e)}")
                
        print("检测线程结束")

    # 启动检测线程
    def _start_detection_thread(self):
        """启动检测线程"""
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.detection_running = True
            self.detection_thread = threading.Thread(target=self._detection_thread_worker, daemon=True)
            self.detection_thread.start()
            print("检测线程已启动")

    # 停止检测线程
    def _stop_detection_thread(self):
        """停止检测线程"""
        if self.detection_running:
            self.detection_running = False
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=2)
            print("检测线程已停止")
    def plan_paths(self, command: str = "") -> bool:
        """路径规划"""
        try:
            self._update_state(SystemState.PLANNING, "正在进行路径规划...")
            
            self.load_json_file(file_path="detection_results/detection_results.json")

            # 创建路径规划对象
            obj = algorithms.PathPlanner(self.grid_vehicles, self.grid_obstacles, self.grid_destinations)

            # 调用LLM (可选)
            function_list = test.call_LLM(self.grid_vehicles, self.grid_destinations, command)
            print
            if function_list is None:
                self._report_error("LLM调用失败")
                return False
            else:
                # 解释LLM结果并生成路径
                self.grid_path_results = test.Interpret_function_list(function_list, obj)

                if not self.grid_path_results:
                    self._report_error("路径规划失败")
                    return False
                
                # 将栅格路径映射为实际坐标路径
                for i, path_dict in enumerate(self.grid_path_results):
                    for car_id, grid_path in path_dict.items():
                        # grid_path = self.optimize_path(grid_path)
                        img_path = batch_convert_to_image_coordinates(grid_path)
                        real_path = self.mapper.batch_map_to_real_coords(img_path)
                        self.path_results.append({self.car_id_mapping.get(car_id):real_path})

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
                    lookahead_distance=80,
                    max_speed=100.0,
                    min_speed=20.0,
                    goal_tolerance=30,
                    max_angle_control=60.0,
                    turn_in_place_threshold=15.0,
                    bias=car_bias,
                    name=f"vehicle_{vehicle_id}"
                )
                self.trajectory_planners[vehicle_id] = planner

                # 创建状态接收器
                receiver = StateReceiver()
                self.state_receivers[vehicle_id] = receiver

                # 订阅 ROS 话题
                topic_index = vehicle_id
                if topic_index < len(self.car_topics):
                    pose_topic = self.car_topics[topic_index]
                    twist_topic = pose_topic.replace('/pose', '/twist')

                    rospy.Subscriber(pose_topic, PoseStamped, receiver.pose_callback, queue_size=1)
                    rospy.Subscriber(twist_topic, TwistStamped, receiver.twist_callback, queue_size=1)

    def get_actual_trajectory(self, vehicle_id: int) -> List[Tuple[float, float]]:
        controller = self.trajectory_planners.get(vehicle_id)
        if controller:
            return controller.get_trajectory()
        return []
    
    def execute_mission(self) -> bool:
        """执行任务"""
        try:
            self._update_state(SystemState.EXECUTING, "正在执行任务...")
            
            if not self.path_results:
                self._report_error("没有可执行的路径")
                return False
            
            # 启动检测线程
            self._start_detection_thread()
            
            rate = rospy.Rate(10)  # 10Hz控制频率
                
            # 任务执行主循环
            while not self._stop_event.is_set() and not rospy.is_shutdown():
                # 移除了 self._update_vehicle_positions() 调用
                
                self._execute_trajectory_control()

                # 检查任务完成状态
                if self._check_mission_completion():
                    self._update_state(SystemState.COMPLETED, "任务完成")
                    break
                    
                rate.sleep()
            
            # 停止检测线程
            self._stop_detection_thread()
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
            
            # 5. 使用 ROS Timer 稍后执行（在主线程）
            rospy.Timer(rospy.Duration(0.1), self._execute_mission_callback, oneshot=True)
            
            return True  # 立即返回 True
            
        except Exception as e:
            self._report_error(f"任务执行异常: {str(e)}")
            return False

    def _execute_mission_callback(self, event):
        """ROS Timer 回调，执行任务"""
        try:
            self.execute_mission()
        except Exception as e:
            self._report_error(f"任务执行异常: {str(e)}")
        finally:
            self.running = False

    def stop_mission(self):
        """停止任务"""
        self._stop_event.set()
        self._stop_detection_thread()  # 添加这行
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
        # 停止检测线程
        self._stop_detection_thread()  # 添加这行
        
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
        
    def _match_vehicles_by_position(self,
                                 image_real_coords: List[Tuple[float, float]],
                                 ros_coords: List[Tuple[float, float]]) -> Dict[int, int]:
        """
        通过距离最小原则，匹配图像检测到的小车 ID 与 ROS 小车 ID。
        返回映射关系：{检测到的ID: ROS ID}
        """
        mapping = {}
        used_ros = set()

        for img_id, img_coord in enumerate(image_real_coords):
            min_dist = float('inf')
            matched_ros_id = -1
            for ros_id, ros_coord in enumerate(ros_coords):
                if ros_id in used_ros:
                    continue
                dist = np.linalg.norm(np.array(img_coord) - np.array(ros_coord))
                if dist < min_dist:
                    min_dist = dist
                    matched_ros_id = ros_id
            if matched_ros_id != -1:
                mapping[img_id] = matched_ros_id
                used_ros.add(matched_ros_id)

        return mapping        
    
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
            # 1. 拍摄当前图像
            timestamp = int(time.time())
            image_path = f"{self.capture_dir}/capture_{timestamp}.jpg"
            self.camera.capture_rotated_image(file_path=image_path, angle=self.camera_rotation)

            # 2. YOLO检测
            results = predict.detect_objects(path=image_path, show_results=False, verbose=False)
            predict.save_detection_results(results, save_dir="detection_results", verbose=False)

            if not results or not results[0]:
                print("未检测到小车，等待下一次检测...")
                return True

            # 3. 提取图像中心坐标
            vehicle_img_coords = results[0].get('all_vehicles', [])
            if not vehicle_img_coords:
                print("未检测到小车，等待下一次检测...")
                return True

            vehicle_img_coords = [self.get_center_from_bbox(bbox) for bbox in vehicle_img_coords]

            # 4. 映射为实际坐标（图像到实际世界）
            mapped_real_coords = [self.mapper.map_to_real_coords(coord) for coord in vehicle_img_coords]
            
            # 5. 获取 ROS 实时坐标
            vehicle_ros_coords = []
            for topic in self.car_topics:
                real_pos = self._get_car_position(topic)
                if real_pos is None:
                    return False
                vehicle_ros_coords.append(real_pos)

            # 6. 匹配图像ID → ROS小车ID，更新 car_id_mapping
            self.car_id_mapping = self._match_vehicles_by_position(mapped_real_coords, vehicle_ros_coords)

            # 7. 更新 self.vehicles[] 信息
            for img_id, img_coord in enumerate(vehicle_img_coords):
                if img_id in self.car_id_mapping:
                    ros_id = self.car_id_mapping[img_id]
                    if ros_id < len(self.vehicles):
                        self.vehicles[ros_id]['img_coord'] = img_coord
                        self.vehicles[ros_id]['real_coord'] = mapped_real_coords[img_id]
                        self.vehicles[ros_id]['ros_coord'] = vehicle_ros_coords[ros_id]

            # 8. 打印误差信息
            for img_id, real_coord in enumerate(mapped_real_coords):
                if img_id in self.car_id_mapping:
                    ros_id = self.car_id_mapping[img_id]
                    ros_coord = vehicle_ros_coords[ros_id]
                    error = np.linalg.norm(np.array(real_coord) - np.array(ros_coord))
                    print(f"小车（图像ID:{img_id} → ROS ID:{ros_id}）: 映射坐标={real_coord}, ROS坐标={ros_coord}, 误差={error:.2f}mm")

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
        if system.start_mission("小车0清扫目的地0"):
            print("任务启动成功")
            rospy.spin()
            # 等待任务完成或用户中断
            while system.running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("用户中断")
        
    finally:
        system.cleanup()