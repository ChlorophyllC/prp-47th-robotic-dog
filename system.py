from camera import HikvisionCamera as Camera
from coordinate_mapper import CoordinateMapper
import test
import algorithms
import os
import time
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from typing import Tuple, Dict, List, Optional, Callable, Any
import predict
import threading
from enum import Enum
import json
from controller import TrajectoryPlanner, StateReceiver, send_ctrl
from predict import batch_convert_to_image_coordinates, batch_convert_to_grid_coordinates
import itertools
import math
import cv2

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
        # 手动映射（来自 GUI 映射窗口）：{yolo_index(检测序号): real_vehicle_id(ROS vehicle_x)}
        self._manual_car_id_mapping: Optional[Dict[int, int]] = None
        self.grid_path_results = []
        self.path_results = []  # 路径规划结果
        
        # 设置小车配置
        self._setup_vehicle_config(vehicle_ids, car_ips, car_bias, car_port)
        
        # 控制参数
        self.camera_rotation = camera_rotation  # 相机旋转角度
        self.mapper_file = os.path.join(os.path.dirname(__file__), "coordinate_mapper.pkl")

        # 是否强制重建坐标映射器（由 GUI 选择；仅对下一次 setup_coordinate_mapping 生效）
        self.force_rebuild_mapper = False
        
        # 回调函数
        self.status_callback = None  # 状态更新回调
        self.error_callback = None  # 错误回调
        self.progress_callback = None  # 进度回调
        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None  # 事件回调（事件驱动）
        
        # 线程控制
        self.running = False
        self.main_thread = None
        self._stop_event = threading.Event()

        # 任务线程/计时器引用，便于 stop/cleanup 可靠回收
        self._mission_thread: Optional[threading.Thread] = None
        self._mission_timer = None
        
        # 添加检测线程控制
        self.detection_thread = None
        self.detection_running = False
        
        # 控制模块相关
        self.trajectory_planners = {}  # 存储每个小车的轨迹规划器
        self.state_receivers = {}      # 存储每个小车的状态接收器

        # 任务完成标志（GUI 侧会读取）
        self.mission_completed = False

        # 记录“初始位置”（ROS坐标），用于一键回归
        # {vehicle_id: (x, y)}
        self.home_positions: Dict[int, Tuple[float, float]] = {}

        # ==================== 实时避障：障碍物快照缓存 ====================
        # 网格坐标系下的障碍物 bbox（四角点形式，与 detection_results.json 一致）
        self._obstacle_snapshot_lock = threading.Lock()
        self._obstacle_snapshot: Dict[str, Any] = {
            "ts": 0.0,
            "grid_obstacles": [],
            "grid_dynamic_obstacles": [],  # 其他小车（动态障碍，bbox列表）
            "grid_vehicles_by_id": {},     # {vehicle_id: (gx, gy)}
            "grid_vehicle_bboxes_by_id": {},  # {vehicle_id: bbox}
        }

        # 实时避障开关（可被任务级别覆盖）
        self.avoidance_enabled: bool = True
        self._avoidance_enabled_override: Optional[bool] = None
        # 车辆在网格坐标系下的近似半径（用于将其他小车膨胀为障碍）
        self.vehicle_obstacle_radius_grid: float = 4.0
        # obstacle 过期时间（秒）：避免旧障碍物长期阻塞
        self.obstacle_snapshot_ttl_s: float = 1.5
        # 检测线程频率（Hz）
        self.detection_hz: float = 10.0

        # ==================== 检测结果/相机帧：内存缓存（减少磁盘I/O） ====================
        self._frame_lock = threading.Lock()
        self._latest_frame_bgr = None  # OpenCV BGR ndarray
        self._latest_frame_ts: float = 0.0

        self._latest_detection_data = None  # (detection_results, (w,h))
        self._latest_grid_results: Dict[str, List[List[List[float]]]] = {
            "all_vehicles": [],
            "obstacle": [],
            "destination": [],
        }

        # 检测结果落盘限频（仍保留 detection_results.json 供规划/调试使用）
        self.detection_json_write_interval_s: float = 1.0
        self._last_detection_json_write_ts: float = 0.0
        self.detection_debug_save_images: bool = False
        self.detection_debug_image_interval_s: float = 2.0
        self._last_detection_image_write_ts: float = 0.0
        # =====================================================================
        # ===============================================================

        # ==================== 实时避障：重规划线程 ====================
        self.avoidance_hz: float = 2.0
        self.avoidance_lookahead_points: int = 20  # 检查未来多少个轨迹点
        self.avoidance_inflation_grid: float = 2.0  # 障碍物膨胀（网格坐标）
        self._avoidance_goal_grid: Dict[int, Tuple[float, float]] = {}
        self._avoidance_thread: Optional[threading.Thread] = None
        self._avoidance_running: bool = False
        # planner 更新互斥：避免避障线程与 10Hz 控制回路并发改轨
        self._planner_lock = threading.Lock()
        # =============================================================

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        """向上层（GUI）发事件。必须是轻量、线程安全（调用方自行在 GUI 主线程处理）。"""
        if not self.event_callback:
            return
        try:
            self.event_callback({"type": event_type, **payload})
        except Exception:
            # 避免回调异常影响控制流程
            pass
    
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
        
        # 生成ROS话题（按 vehicle_ids 顺序）
        self.car_topics = []
        for vehicle_id in vehicle_ids:
            topic = f"/vrpn_client_node/vehicle_{vehicle_id}/pose"
            self.car_topics.append(topic)

        # 建立 vehicle_id -> 索引 映射，便于从 vehicle_id 找到 topic 索引
        self.vehicle_id_to_index = {vehicle_id: i for i, vehicle_id in enumerate(vehicle_ids)}

        # 设置小车IP配置（使用 vehicle_id 作为 key）
        default_ips = ["192.168.1.208", "192.168.1.205", "192.168.1.207"]
        self.car_ips = {}
        for i, vehicle_id in enumerate(vehicle_ids):
            if car_ips is None:
                if i < len(default_ips):
                    self.car_ips[vehicle_id] = default_ips[i]
                else:
                    self.car_ips[vehicle_id] = f"192.168.1.{200 + i}"
            else:
                if vehicle_id in car_ips:
                    self.car_ips[vehicle_id] = car_ips[vehicle_id]
                else:
                    self.car_ips[vehicle_id] = f"192.168.1.{200 + i}"

        # 设置小车偏置配置（使用 vehicle_id 作为 key）
        default_bias = [0, 0, 0]
        self.car_bias = {}
        for i, vehicle_id in enumerate(vehicle_ids):
            if car_bias is None:
                if i < len(default_bias):
                    self.car_bias[vehicle_id] = default_bias[i]
                else:
                    self.car_bias[vehicle_id] = 0
            else:
                if vehicle_id in car_bias:
                    self.car_bias[vehicle_id] = car_bias[vehicle_id]
                else:
                    self.car_bias[vehicle_id] = 0
        
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
        if self.running:
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
                     progress_callback: Callable = None,
                     event_callback: Callable[[Dict[str, Any]], None] = None):
        """设置回调函数（支持事件回调，用于 GUI 事件驱动更新）"""
        self.status_callback = status_callback
        self.error_callback = error_callback
        self.progress_callback = progress_callback
        self.event_callback = event_callback

    def set_manual_vehicle_mapping(self, mapping: Dict[int, int]) -> None:
        """设置手动 YOLO->真实车辆ID 映射。

        mapping: {检测序号(0..n-1): 真实车辆ID(与 self.vehicle_ids/ROS vehicle_x 一致)}
        """
        try:
            cleaned = {int(k): int(v) for k, v in (mapping or {}).items()}
        except Exception:
            cleaned = None
        self._manual_car_id_mapping = cleaned
        if cleaned is not None:
            # 同步到 car_id_mapping，规划阶段直接使用
            self.car_id_mapping = dict(cleaned)
            print(f"使用手动小车ID映射: {self.car_id_mapping}")

    def set_force_rebuild_mapper(self, force: bool) -> None:
        """设置是否强制重建坐标映射器（mapper.pkl）。

        说明：
        - 仅影响下一次 setup_coordinate_mapping()
        - 一般只有在相机位置/高度/焦距/旋转角改变后才需要重建
        """
        try:
            self.force_rebuild_mapper = bool(force)
        except Exception:
            self.force_rebuild_mapper = False

    def prepare_coordinate_mapping_from_manual_mapping(self) -> bool:
        """基于 GUI 的手动 YOLO->真实车辆ID 映射，立即生成/更新坐标映射器(pkl)。

        目的：
        - 用户在 GUI 保存映射后就落盘 coordinate_mapper.pkl
        - 避免等到 start_mission/规划阶段才生成

        约束：
        - 需要 ROS 可用（能读到 vehicle_x 位姿）
        - 需要 detection_results/detection_results.json 存在
        """
        try:
            # 语义：既然用户已经手动标注了三车对应关系，就应该覆盖/重建 mapper.pkl
            prev_force = bool(getattr(self, 'force_rebuild_mapper', False))
            self.force_rebuild_mapper = True

            # 确保 ROS 节点可用（在 GUI 线程/后台线程调用都允许，禁用 signals）
            if not rospy.get_node_uri():
                rospy.init_node('vehicle_control_system_prepare_mapping', anonymous=True, disable_signals=True)

            # 确保 mapper 对象存在（手动映射路径不依赖相机）
            if self.mapper is None:
                self.mapper = CoordinateMapper()

            # 若没有手动映射，直接失败（避免走到需要相机的 fallback 逻辑）
            if not self._manual_car_id_mapping:
                self._report_error("未提供手动车辆映射，无法提前生成坐标映射器")
                return False

            ok = bool(self.setup_coordinate_mapping())
            # 不管成功与否，都恢复原先的 force 标志（setup_coordinate_mapping 成功也会自行复位）
            self.force_rebuild_mapper = prev_force
            return ok
        except Exception as e:
            try:
                self._report_error(f"提前生成坐标映射器失败: {str(e)}")
            except Exception:
                pass
            return False

    def infer_yolo_to_real_mapping_for_gui(self, grid_vehicles: List) -> Dict[int, int]:
        """给 GUI 使用：基于当前检测到的 grid_vehicles，推断 YOLO序号 -> 真实 vehicle_id。

        典型场景：
        - 已经存在 coordinate_mapper.pkl（坐标映射器），GUI 选择不弹出手动映射窗口
        - 但 GUI 仍希望在画面上标注出 V真实(Yi)，便于用户输入命令

        返回：{yolo_index: real_vehicle_id}
        """
        try:
            if not grid_vehicles:
                return {}

            # 确保 ROS 节点可用（禁用 signals，允许从后台线程调用）
            if not rospy.get_node_uri():
                rospy.init_node('vehicle_control_system_infer_mapping', anonymous=True, disable_signals=True)

            # 确保 mapper 可用：优先加载 mapper_file
            if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
                loaded = CoordinateMapper.load_mapper(self.mapper_file)
                if loaded and getattr(loaded, 'is_initialized', False):
                    self.mapper = loaded
            if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
                return {}

            # 获取 ROS 坐标（按 vehicle_ids/topic 顺序）
            vehicle_ros_coords: List[Tuple[float, float]] = []
            for topic in self.car_topics:
                pos = self._get_car_position(topic, timeout=0.8)
                if pos is None:
                    return {}
                vehicle_ros_coords.append(pos)

            # grid bbox -> grid center -> image center -> real center
            grid_centers: List[Tuple[float, float]] = []
            for corners in grid_vehicles:
                try:
                    xs = [float(p[0]) for p in corners]
                    ys = [float(p[1]) for p in corners]
                    grid_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
                except Exception:
                    return {}
            img_centers = batch_convert_to_image_coordinates(grid_centers)
            mapped_real_centers = [self.mapper.map_to_real_coords(pt) for pt in img_centers]

            return self._match_vehicles_by_position(mapped_real_centers, vehicle_ros_coords)
        except Exception:
            return {}

    def reset_mission_state(self) -> None:
        """清空上一轮任务的结果与控制器，保证可重复 start_mission。"""
        # 任务相关状态
        self.mission_completed = False
        self.path_results = []
        self.grid_path_results = []

        # 清理旧的规划器/日志/套接字
        for planner in list(self.trajectory_planners.values()):
            try:
                if hasattr(planner, 'log_file') and planner.log_file:
                    planner.close_log_file()
            except Exception:
                pass
            try:
                if hasattr(planner, 'car_communication') and planner.car_communication:
                    planner.car_communication.close()
            except Exception:
                pass
        self.trajectory_planners.clear()
        self.state_receivers.clear()

        # 注意：不清空 _manual_car_id_mapping / car_id_mapping
        # 手动映射属于用户配置，应跨任务保留

    def _should_abort(self) -> bool:
        return self._stop_event.is_set() or rospy.is_shutdown()
        
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
        self._emit_event("state", state=new_state.value, message=message)
        print(f"状态更新: {new_state.value} - {message}")
        
    def _report_error(self, error_msg: str):
        """报告错误"""
        self._update_state(SystemState.ERROR, error_msg)
        if self.error_callback:
            self.error_callback(error_msg)
        self._emit_event("error", message=error_msg)
        print(f"错误: {error_msg}")
        
    def _report_progress(self, progress: float, message: str = ""):
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(progress, message)
        self._emit_event("progress", progress=progress, message=message)
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

            # 尝试加载现有映射器（除非要求强制重建）
            if not getattr(self, 'force_rebuild_mapper', False):
                loaded_mapper = CoordinateMapper.load_mapper(self.mapper_file)
                if loaded_mapper:
                    # 仅做提示，不再自动丢弃重建：是否重建由 force_rebuild_mapper 决定
                    try:
                        saved_rotation = getattr(loaded_mapper, 'meta', {}).get('camera_rotation')
                    except Exception:
                        saved_rotation = None
                    if saved_rotation is not None and int(saved_rotation) != int(self.camera_rotation):
                        print(
                            f"⚠ 坐标映射器文件camera_rotation={saved_rotation} 与当前={self.camera_rotation} 不一致。"
                            "如需重建请在GUI勾选‘强制重建坐标映射(pkl)’。"
                        )
                    self.mapper = loaded_mapper
                    print("已加载坐标映射器")
                    return True
            else:
                print("已选择强制重建坐标映射器：将忽略现有 mapper.pkl")
                
            print("未找到坐标映射器文件，正在初始化...")

            # 优先：若 GUI 已提供“你选好的三辆车”的映射（检测序号->真实vehicle_id），
            # 则用 detection_results.json 中对应车辆框的中心点来建立三点仿射映射。
            # 这样三对点的对应关系是确定的，不会受 detect_vehicle 内部排序/截断影响。
            try:
                if self._manual_car_id_mapping and os.path.exists("detection_results/detection_results.json"):
                    with open("detection_results/detection_results.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    grid_vehicles = data.get("all_vehicles", []) or []

                    # CoordinateMapper 需要 3 个点
                    selected_vehicle_ids = list(getattr(self, 'vehicle_ids', []) or [])[:3]
                    if len(selected_vehicle_ids) != 3:
                        raise ValueError("vehicle_ids 不足 3 个，无法建立仿射映射")

                    # 反向索引：真实vehicle_id -> 检测序号
                    real_to_yolo = {int(v): int(k) for k, v in self._manual_car_id_mapping.items()}

                    # 取三辆车对应的 grid bbox 中心（grid coords）
                    grid_centers = []
                    for vid in selected_vehicle_ids:
                        if int(vid) not in real_to_yolo:
                            raise ValueError(f"手动映射缺少 vehicle_{vid} 的对应检测序号")
                        yolo_idx = int(real_to_yolo[int(vid)])
                        if yolo_idx < 0 or yolo_idx >= len(grid_vehicles):
                            raise ValueError(f"检测序号 {yolo_idx} 超出 detection_results.json 车辆数量范围")
                        corners = grid_vehicles[yolo_idx]
                        xs = [p[0] for p in corners]
                        ys = [p[1] for p in corners]
                        grid_centers.append((float(sum(xs)) / len(xs), float(sum(ys)) / len(ys)))

                    # grid center -> image(pixel) center
                    img_centers = batch_convert_to_image_coordinates(grid_centers)

                    # 获取这三辆车的 ROS 坐标（按 selected_vehicle_ids 顺序）
                    real_coords = []
                    for vid in selected_vehicle_ids:
                        topic_index = self.vehicle_id_to_index.get(int(vid))
                        if topic_index is None or topic_index >= len(self.car_topics):
                            raise ValueError(f"vehicle_{vid} 未配置对应 ROS topic")
                        pos = self._get_car_position(self.car_topics[topic_index])
                        if pos is None:
                            raise ValueError(f"无法获取 vehicle_{vid} 的 ROS 坐标")
                        real_coords.append(pos)

                    # 初始化映射
                    if self.mapper.initialize_transform(img_centers, real_coords):
                        try:
                            if not hasattr(self.mapper, 'meta') or self.mapper.meta is None:
                                self.mapper.meta = {}
                            self.mapper.meta['camera_rotation'] = int(self.camera_rotation)
                            self.mapper.meta['vehicle_ids_for_mapping'] = [int(x) for x in selected_vehicle_ids]
                        except Exception:
                            pass
                        self.mapper.save_mapper(self.mapper_file)
                        print("坐标映射初始化完成（使用手动选定的三辆车）")
                        # 本次已按需重建，自动复位开关
                        self.force_rebuild_mapper = False
                        return True
            except Exception as e:
                # 手动三车初始化失败则回退到旧逻辑
                print(f"使用手动三车初始化坐标映射失败，回退自动检测方式：{str(e)}")
            
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
                try:
                    if not hasattr(self.mapper, 'meta') or self.mapper.meta is None:
                        self.mapper.meta = {}
                    self.mapper.meta['camera_rotation'] = int(self.camera_rotation)
                except Exception:
                    pass
                self.mapper.save_mapper(self.mapper_file)
                print("坐标映射初始化完成")
                # 本次已按需重建，自动复位开关
                self.force_rebuild_mapper = False
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
                vid = self.vehicle_ids[i] if i < len(self.vehicle_ids) else i
                self.vehicles.append({
                    'id': vid,
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
        """加载JSON文件（系统侧仅做数据加载，不做任何 UI 操作）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if self.validate_json_data(data):
                self.grid_vehicles = data.get("all_vehicles", [])
                self.grid_obstacles = data.get("obstacle", [])
                self.grid_destinations = data.get("destination", [])
                return True
            else:
                raise ValueError("JSON格式不正确")
                
        except Exception as e:
            self._report_error(f"加载JSON失败: {str(e)}")
        return False
    
    def validate_json_data(self, data):
        """验证JSON数据格式"""
        required_keys = ["all_vehicles", "obstacle", "destination"]
        return all(key in data for key in required_keys)

    def _reorder_grid_vehicles_to_real_order(self) -> bool:
        """在规划阶段将 self.grid_vehicles 重排成“真实车辆顺序”。

        目标：让 LLM/算法层使用的车辆下标(小车0/1/2...) 直接对应真实 vehicle_id 的顺序
        （即 self.vehicle_ids 的顺序），从而用户命令里写“车辆0/2”就是 ROS vehicle_0/2。

        说明：
        - detection_results.json 中的 all_vehicles 是网格坐标角点（非像素坐标）
        - 需要：网格中心 -> 像素中心 -> 坐标映射器 -> 真实坐标，与 ROS 坐标做最近距离匹配
        - 若 GUI 提供了手动映射（检测序号->真实 vehicle_id），会优先覆盖自动匹配结果
        """
        if not self.grid_vehicles:
            self._report_error("规划数据中未包含车辆")
            return False
        if not getattr(self, 'vehicle_ids', None):
            self._report_error("vehicle_ids 未配置，无法对齐真实车辆顺序")
            return False

        # 规划阶段优先尝试 ROS+mapper 自动匹配（无需用户重新映射 YOLO 序号）
        # 若 ROS 暂不可用/mapper 未就绪，再回退到手动映射。
        if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
            # mapper 不可用时，只能依赖手动映射
            mapping: Dict[int, int] = {}
        else:
            # 1) 获取 ROS 坐标（按 self.vehicle_ids/self.car_topics 的顺序）
            vehicle_ros_coords: List[Tuple[float, float]] = []
            ros_ok = True
            for topic in self.car_topics:
                # 规划阶段不应长时间卡住：给较短超时；失败则回退手动
                real_pos = self._get_car_position(topic, timeout=1.0)
                if real_pos is None:
                    ros_ok = False
                    break
                vehicle_ros_coords.append(real_pos)

            mapping = {}
            if ros_ok and vehicle_ros_coords:
                # 2) 将 grid_vehicles（四角点）转换为真实坐标中心点
                try:
                    grid_centers: List[Tuple[float, float]] = []
                    for corners in self.grid_vehicles:
                        xs = [p[0] for p in corners]
                        ys = [p[1] for p in corners]
                        grid_centers.append((float(sum(xs)) / len(xs), float(sum(ys)) / len(ys)))

                    img_centers = batch_convert_to_image_coordinates(grid_centers)
                    mapped_real_centers = [self.mapper.map_to_real_coords(pt) for pt in img_centers]
                except Exception as e:
                    self._report_error(f"车辆坐标转换失败: {str(e)}")
                    return False

                # 3) 自动匹配：检测序号 -> 真实 vehicle_id（全局最优分配，避免贪心错配）
                mapping = self._match_vehicles_by_position(mapped_real_centers, vehicle_ros_coords)

        # 如果自动匹配失败，回退手动映射（仍允许无 ROS 的情况下规划）
        if not mapping and self._manual_car_id_mapping:
            try:
                mapping = {int(k): int(v) for k, v in self._manual_car_id_mapping.items()}
            except Exception:
                mapping = {}

        if not mapping:
            self._report_error("无法建立车辆匹配关系（ROS不可用且无有效手动映射）")
            return False

        # 自动模式下也只保留 controllable vehicles（按 vehicle_ids 顺序）
        desired_vehicle_ids = [int(v) for v in (getattr(self, 'vehicle_ids', []) or [])]
        if not desired_vehicle_ids:
            self._report_error("vehicle_ids 未配置，无法对齐真实车辆顺序")
            return False

        inverse: Dict[int, int] = {}
        for src_idx, vid in mapping.items():
            if vid not in inverse:
                inverse[int(vid)] = int(src_idx)

        missing = [vid for vid in desired_vehicle_ids if vid not in inverse]
        if missing:
            self._report_error(f"自动匹配未覆盖全部车辆: {missing}")
            return False

        reordered = []
        for vid in desired_vehicle_ids:
            src_idx = inverse.get(vid)
            if src_idx is None or src_idx < 0 or src_idx >= len(self.grid_vehicles):
                self._report_error(f"自动匹配得到的检测序号 {src_idx} 超出车辆列表范围")
                return False
            reordered.append(self.grid_vehicles[src_idx])

        self.grid_vehicles = reordered
        self.car_id_mapping = {i: int(desired_vehicle_ids[i]) for i in range(len(self.grid_vehicles))}
        print(f"规划阶段车辆顺序已对齐(自动/回退匹配): car_id_mapping={self.car_id_mapping}")
        return True

    def _refresh_grid_vehicle_positions_from_ros(self) -> bool:
        """用当前 ROS 位姿刷新 grid_vehicles 的矩形位置（不需要重新检测/重新标定）。

        背景：detection_results.json 里的车辆框是“拍照那一刻”的位置。
        第一轮任务结束后车辆已移动，如果下一轮规划继续用旧框，会导致“映射/起点不对”。

        做法：
        - 车辆顺序已在 _reorder_grid_vehicles_to_real_order 对齐到 vehicle_ids 顺序
        - 读取每辆车当前 ROS (x,y)
        - 通过 mapper(real->image) + grid 变换得到当前 grid center
        - 将原矩形按中心平移到新中心（保持尺寸/形状）
        """
        if not self.grid_vehicles:
            return False
        if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
            return False
        if not getattr(self, 'vehicle_ids', None):
            return False

        desired_vehicle_ids = [int(v) for v in (self.vehicle_ids or [])]
        if not desired_vehicle_ids:
            return False

        count = min(len(self.grid_vehicles), len(desired_vehicle_ids))
        desired_vehicle_ids = desired_vehicle_ids[:count]

        # 取 ROS 坐标（短超时，避免卡死）
        ros_coords: List[Tuple[float, float]] = []
        for vid in desired_vehicle_ids:
            idx = self.vehicle_id_to_index.get(int(vid))
            if idx is None or idx >= len(self.car_topics):
                return False
            pos = self._get_car_position(self.car_topics[idx], timeout=1.0)
            if pos is None:
                return False
            ros_coords.append(pos)

        try:
            img_points = self.mapper.batch_map_to_image_coords(ros_coords)
            grid_centers = batch_convert_to_grid_coordinates(img_points)
        except Exception:
            return False

        new_grid_vehicles = []
        for i in range(count):
            template = self.grid_vehicles[i]
            if not template or len(template) < 4:
                new_grid_vehicles.append(template)
                continue

            try:
                xs = [float(p[0]) for p in template]
                ys = [float(p[1]) for p in template]
                old_center = (sum(xs) / len(xs), sum(ys) / len(ys))
                new_center = (float(grid_centers[i][0]), float(grid_centers[i][1]))
                dx = new_center[0] - old_center[0]
                dy = new_center[1] - old_center[1]
                shifted = [(float(p[0]) + dx, float(p[1]) + dy) for p in template]
                new_grid_vehicles.append(shifted)
            except Exception:
                new_grid_vehicles.append(template)

        self.grid_vehicles = new_grid_vehicles
        return True
    
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
        
        period = 0.1
        try:
            if float(getattr(self, 'detection_hz', 10.0)) > 0:
                period = 1.0 / float(getattr(self, 'detection_hz', 10.0))
        except Exception:
            period = 0.1

        while self.detection_running and not self._stop_event.is_set():
            try:
                # 执行检测和更新
                self._update_vehicle_positions()
                # 同步更新避障所需的障碍物快照（YOLO obstacle + 其他小车）
                try:
                    self._update_obstacle_snapshot()
                except Exception:
                    pass
                
                # 等待下一次检测周期
                if not self._stop_event.wait(period):  # 使用Event的wait方法，支持中断
                    continue
                else:
                    break  # 收到停止信号
                    
            except Exception as e:
                print(f"检测线程异常: {str(e)}")
                
        print("检测线程结束")

    def _bbox_from_center_radius_grid(self, center: Tuple[float, float], r: float) -> List[Tuple[float, float]]:
        cx, cy = float(center[0]), float(center[1])
        rr = float(max(r, 0.0))
        return [(cx - rr, cy - rr), (cx - rr, cy + rr), (cx + rr, cy + rr), (cx + rr, cy - rr)]

    def _update_obstacle_snapshot(self) -> None:
        """更新实时避障所需的障碍物快照。

        来源：
        - YOLO obstacle 类（Red/Dog）-> 内存缓存（来自 _update_vehicle_positions）
        - 其他小车（除自己外）作为动态障碍：用 ROS 位姿 -> image -> grid center -> 近似圆转 bbox
        """
        now = time.time()

        grid_obstacles: List[List[Tuple[float, float]]] = []
        try:
            grid_obstacles = list((getattr(self, '_latest_grid_results', {}) or {}).get('obstacle') or [])
        except Exception:
            grid_obstacles = []

        # 动态障碍：其他小车（按 vehicle_id）
        grid_dynamic: List[List[Tuple[float, float]]] = []
        grid_vehicles_by_id: Dict[int, Tuple[float, float]] = {}
        grid_vehicle_bboxes_by_id: Dict[int, List[Tuple[float, float]]] = {}
        try:
            if self.mapper and getattr(self.mapper, 'is_initialized', False) and getattr(self, 'vehicle_ids', None):
                # 优先使用订阅到的 state_receivers（避免频繁 wait_for_message 阻塞）
                ros_coords: List[Tuple[float, float]] = []
                vids: List[int] = []
                for vid in (self.vehicle_ids or []):
                    vid_int = int(vid)
                    receiver = self.state_receivers.get(vid_int)
                    if receiver is not None:
                        pos, _, _ = receiver.get_state()
                        # pos 是 np.array([x,y])
                        ros_coords.append((float(pos[0]), float(pos[1])))
                        vids.append(vid_int)
                        continue

                    # 回退：没有 receiver 时再用 wait_for_message（短超时）
                    idx = self.vehicle_id_to_index.get(vid_int)
                    if idx is None or idx >= len(self.car_topics):
                        continue
                    pos2 = self._get_car_position(self.car_topics[idx], timeout=0.2)
                    if pos2 is None:
                        continue
                    ros_coords.append(pos2)
                    vids.append(vid_int)

                if ros_coords:
                    img_points = self.mapper.batch_map_to_image_coords(ros_coords)
                    grid_centers = batch_convert_to_grid_coordinates(img_points)
                    r = float(getattr(self, 'vehicle_obstacle_radius_grid', 4.0))
                    for idx, (gx, gy) in enumerate(grid_centers):
                        vid_i = int(vids[idx])
                        center = (float(gx), float(gy))
                        bbox = self._bbox_from_center_radius_grid(center, r)
                        grid_vehicles_by_id[vid_i] = center
                        grid_vehicle_bboxes_by_id[vid_i] = bbox
                        grid_dynamic.append(bbox)
        except Exception:
            grid_dynamic = []

        with self._obstacle_snapshot_lock:
            self._obstacle_snapshot = {
                "ts": float(now),
                "grid_obstacles": grid_obstacles,
                "grid_dynamic_obstacles": grid_dynamic,
                "grid_vehicles_by_id": grid_vehicles_by_id,
                "grid_vehicle_bboxes_by_id": grid_vehicle_bboxes_by_id,
            }

    def get_obstacle_snapshot(self) -> Dict[str, Any]:
        """获取最新障碍物快照（线程安全）。"""
        with self._obstacle_snapshot_lock:
            return dict(self._obstacle_snapshot)

    def _start_avoidance_thread(self) -> None:
        enabled = bool(getattr(self, 'avoidance_enabled', True))
        try:
            if getattr(self, '_avoidance_enabled_override', None) is not None:
                enabled = bool(self._avoidance_enabled_override)
        except Exception:
            pass
        if not enabled:
            return
        if self._avoidance_thread is not None and self._avoidance_thread.is_alive():
            return
        self._avoidance_running = True
        self._avoidance_thread = threading.Thread(target=self._avoidance_thread_worker, daemon=True)
        self._avoidance_thread.start()
        self._emit_event("state", state=self.state.value, message="实时避障线程已启动")

    def _stop_avoidance_thread(self) -> None:
        self._avoidance_running = False
        if self._avoidance_thread and self._avoidance_thread.is_alive():
            self._avoidance_thread.join(timeout=2)
        self._avoidance_thread = None

    def _grid_point_collides(self, pt: Tuple[float, float], obstacles: List[List[Tuple[float, float]]], buffer: float) -> bool:
        """点-矩形距离判定（buffer 内视为碰撞）。"""
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            return False
        for rect in obstacles:
            try:
                if algorithms.PathPlanner.point_to_rect_distance((x, y), rect) < float(buffer):
                    return True
            except Exception:
                continue
        return False

    def _real_path_to_grid_path(self, real_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not real_path:
            return []
        if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
            return []
        try:
            img_pts = self.mapper.batch_map_to_image_coords(real_path)
            return batch_convert_to_grid_coordinates(img_pts)
        except Exception:
            return []

    def _grid_path_to_real_path(self, grid_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not grid_path:
            return []
        if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
            return []
        try:
            img_pts = batch_convert_to_image_coordinates(grid_path)
            return self.mapper.batch_map_to_real_coords(img_pts)
        except Exception:
            return []

    def _get_goal_grid_for_vehicle(self, vehicle_id: int) -> Optional[Tuple[float, float]]:
        vid = int(vehicle_id)
        g = self._avoidance_goal_grid.get(vid)
        if g is not None:
            return g
        # 回退：从当前 planner 的最后一个 real 点反算为 grid
        planner = self.trajectory_planners.get(vid)
        if not planner:
            return None
        try:
            last_real = planner.trajectory_points[-1]
        except Exception:
            return None
        grid_pts = self._real_path_to_grid_path([last_real])
        if grid_pts:
            return (float(grid_pts[0][0]), float(grid_pts[0][1]))
        return None

    def _infer_destination_index_from_goal_grid(self, goal_grid: Tuple[float, float], max_dist: float = 25.0) -> Optional[int]:
        """根据 goal_grid 推断其对应的 destination_index。

        目的：实时避障重规划时复用 algorithms.PathPlanner.get_all_obstacles() 逻辑，
        让“除目标外的其他目的地”也作为障碍物参与碰撞判断（包围任务尤其重要）。
        """
        try:
            if not goal_grid:
                return None
            gx, gy = float(goal_grid[0]), float(goal_grid[1])
        except Exception:
            return None

        destinations = getattr(self, 'grid_destinations', None) or []
        if not destinations:
            return None

        best_i = None
        best_d = None
        for i, rect in enumerate(destinations):
            if not rect or len(rect) < 4:
                continue
            try:
                cx, cy = algorithms.PathPlanner.get_rect_center(rect)
                d = float(math.hypot(float(cx) - gx, float(cy) - gy))
            except Exception:
                continue
            if best_d is None or d < best_d:
                best_d = d
                best_i = int(i)

        if best_i is None:
            return None
        if best_d is not None and float(best_d) > float(max_dist):
            # 太远可能不是某个目的地（例如目标点是“绕边角点”），仍返回最近者但降低误判风险
            return best_i
        return best_i

    def _find_destination_index_containing_point(self, pt: Tuple[float, float], padding: float = 0.0) -> Optional[int]:
        """判断 pt 是否落在某个目的地矩形内部（AABB）。

        规则：
        - goal 在某目的地内部：a_star 排除该目的地（允许进入到达）
        - goal 不在任何目的地内部（如包围任务的环绕点）：destination_index=None，
          让“包括目标目的地在内的所有目的地”都算障碍（不穿越目的地区域）
        """
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            return None

        destinations = getattr(self, 'grid_destinations', None) or []
        if not destinations:
            return None

        pad = float(padding)
        for i, rect in enumerate(destinations):
            if not rect or len(rect) < 4:
                continue
            try:
                xs = [float(p[0]) for p in rect]
                ys = [float(p[1]) for p in rect]
                if (min(xs) - pad) <= x <= (max(xs) + pad) and (min(ys) - pad) <= y <= (max(ys) + pad):
                    return int(i)
            except Exception:
                continue

        return None

    def _avoidance_thread_worker(self) -> None:
        """执行期实时避障：检测前方碰撞 -> A* 重规划 -> 热更新 planner 轨迹。"""
        period = 0.5
        try:
            hz = float(getattr(self, 'avoidance_hz', 2.0))
            if hz > 0:
                period = 1.0 / hz
        except Exception:
            period = 0.5

        while self._avoidance_running and not self._stop_event.is_set():
            try:
                # 仅在执行期工作
                if not self.running or self.state != SystemState.EXECUTING:
                    time.sleep(period)
                    continue
                if not self.mapper or not getattr(self.mapper, 'is_initialized', False):
                    time.sleep(period)
                    continue

                snap = self.get_obstacle_snapshot()
                ts = float(snap.get('ts') or 0.0)
                if (time.time() - ts) > float(getattr(self, 'obstacle_snapshot_ttl_s', 1.5)):
                    time.sleep(period)
                    continue

                static_obs = list(snap.get('grid_obstacles') or [])
                dyn_bboxes_by_id: Dict[int, List[Tuple[float, float]]] = dict(snap.get('grid_vehicle_bboxes_by_id') or {})
                dyn_obs_all = list(snap.get('grid_dynamic_obstacles') or [])

                # 对每辆车检查前方路径是否会碰撞
                for vehicle_id, planner in list(self.trajectory_planners.items()):
                    if self._stop_event.is_set() or (not self._avoidance_running):
                        break
                    if not planner or getattr(planner, 'is_finished', False):
                        continue

                    # 取未来若干个 real 轨迹点
                    try:
                        start_idx = int(getattr(planner, 'current_target_idx', 0))
                        future_real = list(planner.trajectory_points[start_idx:start_idx + int(getattr(self, 'avoidance_lookahead_points', 20))])
                    except Exception:
                        future_real = []

                    future_grid = self._real_path_to_grid_path(future_real)
                    if not future_grid:
                        continue

                    # 构造该车的障碍集合：静态 + 其他车
                    vid_int = int(vehicle_id)
                    other_dyn = []
                    if dyn_bboxes_by_id:
                        for other_vid, bbox in dyn_bboxes_by_id.items():
                            if int(other_vid) == vid_int:
                                continue
                            other_dyn.append(bbox)
                    else:
                        # 回退：若无按ID信息，至少剔除不了 self，采用全量（保守）
                        other_dyn = list(dyn_obs_all)

                    obstacles = static_obs + other_dyn
                    buffer = float(getattr(self, 'vehicle_obstacle_radius_grid', 4.0)) + float(getattr(self, 'avoidance_inflation_grid', 2.0))

                    need_replan = False
                    hit_point = None
                    hit_index = None
                    for pt in future_grid:
                        if self._grid_point_collides(pt, obstacles, buffer):
                            need_replan = True
                            try:
                                hit_point = (float(pt[0]), float(pt[1]))
                            except Exception:
                                hit_point = None
                            try:
                                hit_index = int(future_grid.index(pt))
                            except Exception:
                                hit_index = None
                            break

                    if not need_replan:
                        continue

                    goal_grid = self._get_goal_grid_for_vehicle(vid_int)
                    if goal_grid is None:
                        continue

                    # 起点：用最新 grid center（来自 snapshot）
                    grid_center_by_id: Dict[int, Tuple[float, float]] = dict(snap.get('grid_vehicles_by_id') or {})
                    start_center = grid_center_by_id.get(vid_int)
                    if start_center is None:
                        # 回退：使用未来路径第一个 grid 点
                        try:
                            start_center = (float(future_grid[0][0]), float(future_grid[0][1]))
                        except Exception:
                            start_center = None
                    if start_center is None:
                        continue

                    # 构造 A* 输入：vehicles[0] 是自己，其余是其他车
                    r = float(getattr(self, 'vehicle_obstacle_radius_grid', 4.0))
                    self_rect = self._bbox_from_center_radius_grid(start_center, r)
                    other_vehicle_rects = list(other_dyn)
                    vehicles_rects = [self_rect] + other_vehicle_rects

                    destinations = list(getattr(self, 'grid_destinations', None) or [])
                    if destinations:
                        inside_idx = self._find_destination_index_containing_point(goal_grid, padding=0.0)
                        destination_index = int(inside_idx) if inside_idx is not None else None

                        obj = algorithms.PathPlanner(vehicles_rects, static_obs, destinations)
                        new_grid_path = obj.a_star_path_planning(
                            current_vehicle_index=0,
                            destination_index=destination_index,  # 允许 None
                            dict_mode=False,
                            dest_point=(float(goal_grid[0]), float(goal_grid[1])),
                            max_iter=8000,
                        )
                    else:
                        # 回退：没有目的地集合时，仅用一个 goal_rect（兼容特殊目标点）
                        goal_rect = self._bbox_from_center_radius_grid(goal_grid, 1.0)
                        obj = algorithms.PathPlanner(vehicles_rects, static_obs, [goal_rect])
                        new_grid_path = obj.a_star_path_planning(
                            current_vehicle_index=0,
                            destination_index=0,
                            dict_mode=False,
                            dest_point=(float(goal_grid[0]), float(goal_grid[1])),
                            max_iter=8000,
                        )

                    if not new_grid_path:
                        # 无解：保守策略（先降速）；也可进一步扩展为停车
                        try:
                            with self._planner_lock:
                                planner.set_max_speed(max(20.0, planner.max_speed * 0.6))
                        except Exception:
                            pass
                        self._emit_event("state", state=self.state.value, message=f"避障重规划失败，已降速 vehicle_{vid_int}")
                        continue

                    new_real_path = self._grid_path_to_real_path(new_grid_path)
                    if len(new_real_path) < 2:
                        continue

                    # 热更新轨迹（互斥）
                    with self._planner_lock:
                        planner.restore_speed_limits()
                        planner.update_trajectory(new_real_path, reset_pid=True)
                        # 同步 path_results，便于 GUI 后续叠加显示（若有）
                        try:
                            if self.path_results and isinstance(self.path_results[0], dict):
                                self.path_results[0][vid_int] = new_real_path
                        except Exception:
                            pass

                    # 发出结构化事件，便于 GUI 叠加显示避障轨迹/动态障碍
                    try:
                        payload_other_dyn = list(other_dyn) if other_dyn else []
                        self._emit_event(
                            "avoidance_replanned",
                            vehicle_id=int(vid_int),
                            ts=float(time.time()),
                            grid_path=[(float(p[0]), float(p[1])) for p in new_grid_path],
                            start_grid=(float(start_center[0]), float(start_center[1])) if start_center else None,
                            goal_grid=(float(goal_grid[0]), float(goal_grid[1])) if goal_grid else None,
                            hit_point=hit_point,
                            hit_index=hit_index,
                            grid_static_obstacles_count=int(len(static_obs)),
                            grid_other_vehicle_obstacles=payload_other_dyn,
                            buffer=float(buffer),
                        )
                    except Exception:
                        pass

                    self._emit_event("state", state=self.state.value, message=f"避障重规划完成 vehicle_{vid_int}")

                time.sleep(period)
            except Exception:
                time.sleep(period)

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

            # 每次规划前清空上一轮结果，避免累积导致无法进入下一轮
            self.reset_mission_state()

            if self._should_abort():
                return False
            
            if not self.load_json_file(file_path="detection_results/detection_results.json"):
                return False

            # 规划阶段对齐车辆顺序：让 grid_vehicles 的下标与真实 vehicle_ids 顺序一致
            # 从而命令无需做“真实ID<->检测序号”的字符串替换。
            if not self._reorder_grid_vehicles_to_real_order():
                return False

            # 关键修正：用 ROS 当前位姿刷新车辆起点（不需要重新映射坐标，也不要求重新检测）
            # 若 ROS 暂不可用，则保持 detection_results.json 的车辆框作为起点。
            try:
                refreshed = self._refresh_grid_vehicle_positions_from_ros()
                if refreshed:
                    print("规划起点已用 ROS 位姿刷新（grid_vehicles 已平移到当前车位）")
            except Exception:
                pass

            # 创建路径规划对象
            obj = algorithms.PathPlanner(self.grid_vehicles, self.grid_obstacles, self.grid_destinations)

            # 调用LLM (可选)
            function_list = test.call_LLM(self.grid_vehicles, self.grid_destinations, command)
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
                    if self._should_abort():
                        return False
                    for car_id, grid_path in path_dict.items():
                        # grid_path = self.optimize_path(grid_path)
                        img_path = batch_convert_to_image_coordinates(grid_path)
                        real_path = self.mapper.batch_map_to_real_coords(img_path)
                        # car_id 是算法/LLM 使用的“检测序号”(0..n-1)
                        # 需要通过 car_id_mapping 映射到真实 vehicle_id（ROS vehicle_x）
                        mapped_vehicle_id = None
                        try:
                            mapped_vehicle_id = self.car_id_mapping.get(int(car_id))
                        except Exception:
                            mapped_vehicle_id = self.car_id_mapping.get(car_id)

                        # 兜底：如果没有映射，则当作已经是 vehicle_id（允许高级用户直接用真实ID规划）
                        if mapped_vehicle_id is None:
                            try:
                                mapped_vehicle_id = int(car_id)
                            except Exception:
                                mapped_vehicle_id = car_id

                        self.path_results.append({mapped_vehicle_id: real_path})

                        # 保存每辆车的避障目标（网格坐标终点）：用于执行期实时重规划
                        try:
                            if grid_path:
                                last_pt = grid_path[-1]
                                self._avoidance_goal_grid[int(mapped_vehicle_id)] = (float(last_pt[0]), float(last_pt[1]))
                        except Exception:
                            pass

            for i in range(len(self.path_results)):
                if self._should_abort():
                    return False
                self._create_trajectory_planners(task_index=i)
                # 等待任务完成 ...

            print(f"路径规划完成，共{len(self.path_results)}条路径")
            self._emit_event(
                "paths_planned",
                grid_path_results=self.grid_path_results,
                path_results=self.path_results,
            )
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

                # 订阅 ROS 话题：使用 vehicle_id -> index 映射查找 topic
                topic_index = self.vehicle_id_to_index.get(vehicle_id)
                if topic_index is not None and topic_index < len(self.car_topics):
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

            # 启动实时避障线程（低频重规划）
            try:
                self._start_avoidance_thread()
            except Exception:
                pass
            
            rate = rospy.Rate(10)  # 10Hz控制频率
                
            # 任务执行主循环
            while not self._stop_event.is_set() and not rospy.is_shutdown():
                # 移除了 self._update_vehicle_positions() 调用
                
                self._execute_trajectory_control()

                # 检查任务完成状态
                if self._check_mission_completion():
                    self._update_state(SystemState.COMPLETED, "任务完成")
                    self.mission_completed = True
                    self._emit_event("mission_completed")
                    break
                    
                rate.sleep()
            
            # 停止检测线程
            self._stop_detection_thread()
            # 停止避障线程
            try:
                self._stop_avoidance_thread()
            except Exception:
                pass
            return True
            
        except Exception as e:
            self._report_error(f"任务执行失败: {str(e)}")
            return False
        finally:
            # 任务退出时兜底复位 running
            self.running = False
            self._emit_event("running", running=False)
            # 任务级别的避障覆盖只对本次任务生效
            try:
                self._avoidance_enabled_override = None
            except Exception:
                pass
        
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
                    car_ip = self.car_ips.get(vehicle_id)
                    if not car_ip:
                        self._report_error(f"vehicle_{vehicle_id} 未配置IP，无法发送控制")
                        self._stop_event.set()
                        return

                    # 如需排查“走别人的路”，可临时打开此打印：
                    # print(f"发送控制: vehicle_{vehicle_id} -> {car_ip}:{self.car_port}")
                    # 避障线程可能在更新轨迹点，发送控制时加互斥
                    with self._planner_lock:
                        planner.send_control(car_ip, self.car_port)
                except Exception as e:
                    print(f"小车{vehicle_id}控制发送失败: {str(e)}") 

    def start_mission(self, command: str = "") -> bool:
        """启动任务"""
        if self.running:
            self._report_error("任务已在运行中")
            return False
        
        success = False
        try:
            # 启动前先清理上一轮任务残留（允许完成后直接开始下一轮）
            self.reset_mission_state()
            try:
                if self.camera:
                    self.camera.disconnect()
            except Exception:
                pass
            self.camera = None

            self.running = True
            self.mission_completed = False
            self._stop_event.clear()
            self._emit_event("running", running=True)
            
            # 在主线程中初始化ROS节点
            if not rospy.get_node_uri():
                # GUI 会在后台线程调用 start_mission；rospy 默认会注册 signal handler，
                # 这在非主线程会触发：signal only works in main thread。
                # 这里禁用 ROS 自己的 signal 处理，改由上层自行控制退出。
                rospy.init_node('vehicle_control_system', anonymous=True, disable_signals=True)
                print("ROS节点初始化完成")
            
            # 1. 初始化系统（不包括ROS）
            if not self.initialize_system():
                return False

            if self._should_abort():
                return False
                
            # 2. 设置坐标映射
            if not self.setup_coordinate_mapping():
                return False

            if self._should_abort():
                return False
                
            # 3. 若 GUI 提供了手动映射，说明用户已基于 GUI 的检测结果进行了 YOLO->真实ID 标注。
            #    此时不再重复执行系统侧 initial_detection（会生成新的检测序号顺序，反而破坏映射）。
            #    若没有手动映射，则走系统侧检测+自动匹配。
            if not self._manual_car_id_mapping:
                if not self.perform_initial_detection():
                    return False

            if self._should_abort():
                return False
                
            # 4. 路径规划
            if not self.plan_paths(command):
                return False

            if self._should_abort():
                return False

            # 5. 启动任务线程（便于 stop_mission join，避免 Timer 残留）
            self._mission_thread = threading.Thread(target=self.execute_mission, daemon=True)
            self._mission_thread.start()

            success = True
            return True
            
        except Exception as e:
            self._report_error(f"任务执行异常: {str(e)}")
            return False
        finally:
            if not success:
                self._finalize_start_failure()

    def _finalize_start_failure(self):
        """启动失败时的统一收尾（不抛异常）。"""
        try:
            self._update_state(SystemState.IDLE, "任务启动失败，已复位")
            self.running = False
            self._emit_event("running", running=False)
            self._stop_detection_thread()
            self.reset_mission_state()
            try:
                if self.camera:
                    self.camera.disconnect()
            except Exception:
                pass
            self.camera = None
        except Exception:
            pass

    def _execute_mission_callback(self, event):
        """兼容旧逻辑：ROS Timer 回调（当前默认不再使用）。"""
        try:
            self.execute_mission()
        except Exception as e:
            self._report_error(f"任务执行异常: {str(e)}")

    def stop_mission(self):
        """停止任务"""
        # 1) 停止标志
        self._stop_event.set()

        # 1.1) 停止避障线程
        try:
            self._stop_avoidance_thread()
        except Exception:
            pass

        # 2) 停止检测线程
        self._stop_detection_thread()

        # 3) 停止所有小车（尽量保证物理层停下）
        try:
            import socket
            car_communication = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for vehicle_id, car_ip in self.car_ips.items():
                try:
                    send_ctrl(0, 0, car_ip, self.car_port, car_communication)
                except Exception:
                    pass
            try:
                car_communication.close()
            except Exception:
                pass
        except Exception:
            pass

        # 4) 等待任务线程退出
        if self._mission_thread and self._mission_thread.is_alive():
            self._mission_thread.join(timeout=5)

        # 5) 复位状态
        self.running = False
        self.mission_completed = False
        self._update_state(SystemState.IDLE, "任务已停止")
        self._emit_event("mission_stopped")

            
    def pause_mission(self):
        """暂停任务"""
        self._update_state(SystemState.PAUSED, "任务已暂停")
        
    def resume_mission(self):
        """恢复任务"""
        self._update_state(SystemState.EXECUTING, "任务已恢复")
        
    def cleanup(self):
        """清理资源"""
        # 停止任务与检测线程（包含 join）
        try:
            self.stop_mission()
        except Exception:
            pass

        # 停止避障线程（兜底）
        try:
            self._stop_avoidance_thread()
        except Exception:
            pass
        
        # 关闭规划器资源
        for planner in list(self.trajectory_planners.values()):
            try:
                if hasattr(planner, 'log_file') and planner.log_file:
                    planner.close_log_file()
            except Exception:
                pass
            try:
                if hasattr(planner, 'car_communication') and planner.car_communication:
                    planner.car_communication.close()
            except Exception:
                pass
        
        self.reset_mission_state()
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

    def record_home_positions(self, overwrite: bool = False, timeout: float = 2.0, retries: int = 3) -> bool:
        """记录当前 ROS 位姿为“初始位置”。

        overwrite=False 时，如果已记录过则不覆盖。
        """
        try:
            required_ids = [int(v) for v in (self.vehicle_ids or [])]
            if (not overwrite) and self.home_positions:
                # 已有记录但不完整：尝试补齐缺失车辆
                if required_ids and all(int(v) in self.home_positions for v in required_ids):
                    return True

            if not rospy.get_node_uri():
                rospy.init_node('vehicle_control_system_record_home', anonymous=True, disable_signals=True)

            positions: Dict[int, Tuple[float, float]] = dict(self.home_positions) if (not overwrite) else {}
            missing: List[int] = []

            for vehicle_id in required_ids:
                if (not overwrite) and int(vehicle_id) in positions:
                    continue
                idx = self.vehicle_id_to_index.get(int(vehicle_id))
                if idx is None or idx >= len(self.car_topics):
                    missing.append(int(vehicle_id))
                    continue

                got = None
                for _ in range(max(1, int(retries))):
                    pos = self._get_car_position(self.car_topics[idx], timeout=float(timeout))
                    if pos is None:
                        continue
                    try:
                        x, y = float(pos[0]), float(pos[1])
                        if not (math.isfinite(x) and math.isfinite(y)):
                            continue
                        if abs(x) < 1e-6 and abs(y) < 1e-6:
                            continue
                        got = (x, y)
                        break
                    except Exception:
                        continue

                if got is None:
                    missing.append(int(vehicle_id))
                else:
                    positions[int(vehicle_id)] = got

            if required_ids and missing:
                self._emit_event(
                    "state",
                    state=self.state.value,
                    message=f"记录初始位置失败：缺少车辆 {missing} 的ROS位姿（请确认vrpn/话题发布正常）",
                )
                return False

            if not positions:
                return False

            self.home_positions = positions
            print(f"已记录初始位置: {self.home_positions}")
            return True
        except Exception:
            return False

    def start_return_to_home(self) -> bool:
        """启动“一键回归初始位置”。

        做法：
        - 每辆车生成 [当前位姿 -> home位姿] 的直线路径
        - 复用现有 TrajectoryPlanner + execute_mission 控制回路
        """
        if self.running:
            self._report_error("系统正在运行中，请先停止任务再回归初始位置")
            return False

        if not self.home_positions:
            self._report_error("尚未记录初始位置：请先完成一次检测/映射，或确保ROS位姿可用")
            return False

        try:
            # ROS
            if not rospy.get_node_uri():
                rospy.init_node('vehicle_control_system_return_home', anonymous=True, disable_signals=True)

            # 清理上一轮残留
            self.reset_mission_state()
            self._stop_event.clear()
            self.mission_completed = False
            self.running = True
            self._emit_event("running", running=True)

            # 回归初始位置：禁用实时避障，避免重规划导致来回
            self._avoidance_enabled_override = False

            # 构造回归路径（要求所有车辆都能获取到当前位置与home，否则提示）
            paths: Dict[int, List[Tuple[float, float]]] = {}
            missing: List[int] = []
            for vehicle_id in (self.vehicle_ids or []):
                vid = int(vehicle_id)
                home = self.home_positions.get(vid)
                if home is None:
                    missing.append(vid)
                    continue
                idx = self.vehicle_id_to_index.get(vid)
                if idx is None or idx >= len(self.car_topics):
                    missing.append(vid)
                    continue

                cur = None
                for _ in range(2):
                    cur = self._get_car_position(self.car_topics[idx], timeout=2.0)
                    if cur is not None:
                        break
                if cur is None:
                    missing.append(vid)
                    continue
                cur_pt = (float(cur[0]), float(cur[1]))
                home_pt = (float(home[0]), float(home[1]))
                paths[vid] = [cur_pt, home_pt]

            if missing:
                self._emit_event(
                    "state",
                    state=self.state.value,
                    message=f"回归初始位置：车辆 {missing} 缺少home/当前位姿，已跳过（建议重新记录初始位置）",
                )

            if not paths:
                self._report_error("无法生成回归路径：未获取到有效的当前位置/初始位置")
                self.running = False
                self._emit_event("running", running=False)
                return False

            # 用 path_results 复用后续流程
            self.path_results = [paths]

            # 创建控制器并执行
            self._create_trajectory_planners(task_index=0)
            self._mission_thread = threading.Thread(target=self.execute_mission, daemon=True)
            self._mission_thread.start()
            return True
        except Exception as e:
            self._report_error(f"回归初始位置启动失败: {str(e)}")
            try:
                self.running = False
                self._emit_event("running", running=False)
            except Exception:
                pass
            return False
        
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
        # 目标：最小化总距离（全局最优），避免贪心导致的错配
        if not image_real_coords or not ros_coords:
            return {}

        det_pts = [np.array(p, dtype=float) for p in image_real_coords]
        ros_pts = [np.array(p, dtype=float) for p in ros_coords]

        n_det = len(det_pts)
        n_ros = len(ros_pts)
        if n_det == 0 or n_ros == 0:
            return {}

        # 只匹配可控车辆数（ROS数）。若检测多于 ROS，则从检测中选出最合适的那 n_ros 个。
        k = min(n_ros, n_det)

        # 距离矩阵
        dist = np.zeros((n_det, n_ros), dtype=float)
        for i in range(n_det):
            for j in range(n_ros):
                dist[i, j] = float(np.linalg.norm(det_pts[i] - ros_pts[j]))

        best_cost = float('inf')
        best_det_indices = None
        best_perm = None

        # 组合 + 全排列：k<=3 时非常快；k>6 时会爆炸，但本项目通常是 3 辆车
        if k > 6:
            k = 6

        # 选 k 个检测框，与 k 个 ROS 车做一一对应（当 n_det>n_ros 时，这是关键）
        for det_choice in itertools.combinations(range(n_det), k):
            for perm in itertools.permutations(range(n_ros), k):
                cost = 0.0
                for t in range(k):
                    cost += dist[det_choice[t], perm[t]]
                    if cost >= best_cost:
                        break
                if cost < best_cost:
                    best_cost = cost
                    best_det_indices = det_choice
                    best_perm = perm

        if best_det_indices is None or best_perm is None:
            return {}

        mapping: Dict[int, int] = {}
        for t in range(k):
            det_idx = int(best_det_indices[t])
            ros_idx = int(best_perm[t])
            if hasattr(self, 'vehicle_ids') and ros_idx < len(self.vehicle_ids):
                vehicle_id = int(self.vehicle_ids[ros_idx])
            else:
                vehicle_id = ros_idx
            mapping[det_idx] = vehicle_id

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
                # closest_idx 是 ROS 列表中的索引，将其转换为真实 vehicle_id
                if hasattr(self, 'vehicle_ids') and closest_idx < len(self.vehicle_ids):
                    mapped_vehicle_id = self.vehicle_ids[closest_idx]
                else:
                    mapped_vehicle_id = closest_idx
                self.car_id_mapping[i] = mapped_vehicle_id
                
        print(f"小车ID映射: {self.car_id_mapping}")
        
    def _update_vehicle_positions(self) -> bool:
        """更新小车位置"""
        try:
            if self.camera is None:
                return False

            # 1) 内存取帧（不落盘）
            frame = None
            try:
                frame = self.camera.get_frame(timeout=1000, angle=self.camera_rotation)
            except Exception:
                frame = None

            if frame is None:
                print("未获取到相机帧，等待下一次检测...")
                return True

            now = time.time()
            with self._frame_lock:
                self._latest_frame_bgr = frame
                self._latest_frame_ts = float(now)

            # 2) YOLO检测（内存帧推理）
            results = predict.detect_objects_from_frame(frame, show_results=False, verbose=False)
            self._latest_detection_data = results

            # 3) 将检测结果转换为网格四角点（内存缓存，供避障快照使用）
            try:
                self._latest_grid_results = {
                    "all_vehicles": [],
                    "obstacle": [],
                    "destination": [],
                }
                if results is not None:
                    det, (image_width, image_height) = results
                    for category in (det or {}):
                        for bbox in det.get(category, []) or []:
                            x1, y1, x2, y2 = bbox
                            gx1, gy1, gx2, gy2 = predict.convert_to_grid_coordinates(
                                x1, y1, x2, y2,
                                image_width, image_height,
                                grid_width=144, grid_height=108,
                            )
                            corners = predict.bbox_to_corners(gx1, gy1, gx2, gy2)
                            self._latest_grid_results[category].append(corners)
            except Exception:
                pass

            # 4) 可选：限频写入 detection_results.json（兼容旧流程/规划阶段读取）
            try:
                need_write = False
                out_path = os.path.join("detection_results", "detection_results.json")
                if not os.path.exists(out_path):
                    need_write = True
                elif (now - float(self._last_detection_json_write_ts)) >= float(self.detection_json_write_interval_s):
                    need_write = True

                if need_write and results is not None:
                    predict.save_detection_results(results, save_dir="detection_results", verbose=False)
                    self._last_detection_json_write_ts = float(now)
            except Exception:
                pass

            # 5) 可选：限频保存调试帧到 captures（默认关闭）
            try:
                if bool(getattr(self, 'detection_debug_save_images', False)):
                    if (now - float(self._last_detection_image_write_ts)) >= float(getattr(self, 'detection_debug_image_interval_s', 2.0)):
                        ts = int(now)
                        image_path = f"{self.capture_dir}/capture_{ts}.jpg"
                        try:
                            cv2.imwrite(image_path, frame)
                            self._last_detection_image_write_ts = float(now)
                        except Exception:
                            pass
            except Exception:
                pass

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

            # 6. 匹配图像ID → vehicle_id（按“本帧检测顺序”进行就近匹配）
            # 注意：YOLO 输出的检测列表顺序在不同帧可能会交换，所以“图像ID=0/1/2”不是稳定ID。
            # - self._manual_car_id_mapping: 用于“规划阶段”的检测序号->真实ID（来自 GUI 点选），应保持不变
            # - frame_mapping: 用于“本帧更新/日志”的检测序号->真实ID，应每帧重新匹配
            frame_mapping = self._match_vehicles_by_position(mapped_real_coords, vehicle_ros_coords)
            if not self._manual_car_id_mapping:
                # 没有手动映射时，系统用自动匹配结果作为全局映射
                self.car_id_mapping = dict(frame_mapping)

            # 7. 更新 self.vehicles[] 信息（按 vehicle_ids 的顺序）
            for img_id, img_coord in enumerate(vehicle_img_coords):
                if img_id in frame_mapping:
                    mapped_vehicle_id = frame_mapping[img_id]
                    idx = self.vehicle_id_to_index.get(mapped_vehicle_id)
                    if idx is not None and idx < len(self.vehicles):
                        self.vehicles[idx]['img_coord'] = img_coord
                        self.vehicles[idx]['real_coord'] = mapped_real_coords[img_id]
                        self.vehicles[idx]['ros_coord'] = vehicle_ros_coords[idx]

            # 8. 打印误差信息
            for img_id, real_coord in enumerate(mapped_real_coords):
                if img_id in frame_mapping:
                    vehicle_id = frame_mapping[img_id]
                    topic_index = self.vehicle_id_to_index.get(vehicle_id)
                    ros_coord = vehicle_ros_coords[topic_index] if topic_index is not None and topic_index < len(vehicle_ros_coords) else None
                    if ros_coord is None:
                        continue
                    error = np.linalg.norm(np.array(real_coord) - np.array(ros_coord))
                    print(f"小车（本帧图像序号:{img_id} → vehicle_id:{vehicle_id}）: 映射坐标={real_coord}, ROS坐标={ros_coord}, 误差={error:.2f}mm")

            return True

        except Exception as e:
            print(f"更新小车位置失败: {str(e)}")
            return False

    def get_latest_frame(self, max_age_s: float = 2.0, copy: bool = True):
        """获取最新相机帧（OpenCV BGR）。

        供 GUI 背景显示/调试使用。若帧过旧则返回 None。
        """
        try:
            now = time.time()
            with self._frame_lock:
                if self._latest_frame_bgr is None:
                    return None
                if (now - float(self._latest_frame_ts)) > float(max_age_s):
                    return None
                return self._latest_frame_bgr.copy() if copy else self._latest_frame_bgr
        except Exception:
            return None
       
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