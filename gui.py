import json
import os
import time
import threading
from datetime import datetime
import predict 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import matplotlib
from system import VehicleControlSystem
from camera import HikvisionCamera as Camera
from predict import batch_convert_to_image_coordinates
class VehiclePlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆路径规划系统")
        self.root.geometry("1600x1200")
        matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
        
        # 数据存储
        self.grid_vehicles = []
        self.grid_obstacles = []
        self.grid_destinations = []
        
        # 自动检测标志
        self.auto_detect_running = False
        self.detection_thread = None
        self.background = None
        
        # 创建界面
        self.create_widgets()
        
        # 车辆控制系统
        self.vehicle_system = VehicleControlSystem()
        self.setup_system_callbacks()
        
        # 车辆配置参数
        self.vehicle_ids = [1, 2, 3]  # 默认值
        self.car_ips = {0: "192.168.1.208", 1: "192.168.1.205", 2: "192.168.1.207"}  # 默认值
        self.car_bias = {0: -5, 1: 7, 2: 0}  # 默认值
        self.car_port = 12345  # 默认值
        self.camera_rotation = -21
        
        # 首次检测标志
        self.first_detection_done = False

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右侧可视化面板
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 控制面板内容
        self.create_control_panel(control_frame)
        
        # 可视化面板内容
        self.create_visualization_panel(viz_frame)
        
    def create_control_panel(self, parent):
        # 文件导入区域
        file_frame = ttk.LabelFrame(parent, text="文件导入", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="导入JSON文件", 
                  command=self.load_json_file).pack(fill=tk.X)
        
        # 车辆配置区域
        config_frame = ttk.LabelFrame(parent, text="车辆配置", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Vehicle IDs 输入
        ttk.Label(config_frame, text="Vehicle IDs (逗号分隔):").pack(anchor=tk.W)
        self.vehicle_ids_entry = tk.Entry(config_frame, width=35)
        self.vehicle_ids_entry.pack(fill=tk.X, pady=(0, 5))
        self.vehicle_ids_entry.insert(0, "1,2,3")
        
        # Car IPs 输入
        ttk.Label(config_frame, text="Car IPs (格式: id1:ip1,id2:ip2):").pack(anchor=tk.W)
        self.car_ips_entry = tk.Entry(config_frame, width=35)
        self.car_ips_entry.pack(fill=tk.X, pady=(0, 5))
        self.car_ips_entry.insert(0, "1:192.168.1.208,2:192.168.1.205,3:192.168.1.207")
        
        # Car Bias 输入
        ttk.Label(config_frame, text="Car Bias (格式: id1:bias1,id2:bias2):").pack(anchor=tk.W)
        self.car_bias_entry = tk.Entry(config_frame, width=35)
        self.car_bias_entry.pack(fill=tk.X, pady=(0, 5))
        self.car_bias_entry.insert(0, "1:-5,2:7,3:0")
        
        # Car Port 输入
        ttk.Label(config_frame, text="Car Port:").pack(anchor=tk.W)
        self.car_port_entry = tk.Entry(config_frame, width=35)
        self.car_port_entry.pack(fill=tk.X, pady=(0, 5))
        self.car_port_entry.insert(0, "12345")
        
        # 应用配置按钮
        ttk.Button(config_frame, text="应用配置", 
                command=self.apply_vehicle_config).pack(fill=tk.X, pady=(5, 0))
        
        # 首次检测
        detect_frame = ttk.LabelFrame(parent, text="首次检测", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.first_detect_btn = ttk.Button(detect_frame, text="开始首次检测", 
                                        command=self.start_first_detection)
        self.first_detect_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.detection_status = ttk.Label(detect_frame, text="未进行首次检测", 
                                        foreground="orange")
        self.detection_status.pack(pady=(5, 0))
        # 数据显示区域
        data_frame = ttk.LabelFrame(parent, text="数据信息", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.data_text = scrolledtext.ScrolledText(data_frame, height=8, width=35)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        system_frame = ttk.LabelFrame(parent, text="系统控制", padding=10)
        system_frame.pack(fill=tk.X, pady=(0, 10))

        self.stop_mission_btn = ttk.Button(system_frame, text="停止任务", 
                                        command=self.stop_mission, state=tk.DISABLED)
        self.stop_mission_btn.pack(fill=tk.X)

        # 命令输入区域
        command_frame = ttk.LabelFrame(parent, text="命令输入", padding=10)
        command_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(command_frame, text="输入命令:").pack(anchor=tk.W)
        self.command_entry = tk.Text(command_frame, height=3, width=35)
        self.command_entry.pack(fill=tk.X, pady=(5, 0))
        
        # 设置默认命令
        self.command_entry.insert(tk.END, "用车辆0和2包围目的地0。")
        
        # 执行按钮
        ttk.Button(command_frame, text="执行路径规划", 
                  command=self.execute_planning).pack(fill=tk.X, pady=(10, 0))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(parent, text="执行结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=8, width=35)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
    def create_visualization_panel(self, parent):
        # 创建matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化空图
        self.update_visualization()

    def apply_vehicle_config(self):
        """应用车辆配置"""
        try:
            # 解析 vehicle_ids
            vehicle_ids_str = self.vehicle_ids_entry.get().strip()
            if vehicle_ids_str:
                self.vehicle_ids = [int(x.strip()) for x in vehicle_ids_str.split(',')]
            
            # 解析 car_ips
            car_ips_str = self.car_ips_entry.get().strip()
            if car_ips_str:
                car_ips_dict = {}
                for pair in car_ips_str.split(','):
                    if ':' in pair:
                        vehicle_id, ip = pair.split(':', 1)
                        car_ips_dict[int(vehicle_id.strip())] = ip.strip()
                # 转换为索引映射
                self.car_ips = {}
                for i, vehicle_id in enumerate(self.vehicle_ids):
                    if vehicle_id in car_ips_dict:
                        self.car_ips[i] = car_ips_dict[vehicle_id]
                    else:
                        self.car_ips[i] = f"192.168.1.{200 + i}"
            
            # 解析 car_bias
            car_bias_str = self.car_bias_entry.get().strip()
            if car_bias_str:
                car_bias_dict = {}
                for pair in car_bias_str.split(','):
                    if ':' in pair:
                        vehicle_id, bias = pair.split(':', 1)
                        car_bias_dict[int(vehicle_id.strip())] = float(bias.strip())
                # 转换为索引映射
                self.car_bias = {}
                for i, vehicle_id in enumerate(self.vehicle_ids):
                    if vehicle_id in car_bias_dict:
                        self.car_bias[i] = car_bias_dict[vehicle_id]
                    else:
                        self.car_bias[i] = 0
            
            # 解析 car_port
            car_port_str = self.car_port_entry.get().strip()
            if car_port_str:
                self.car_port = int(car_port_str)
            
            # 应用配置到系统
            self.vehicle_system._setup_vehicle_config(
                vehicle_ids=self.vehicle_ids,
                car_ips={self.vehicle_ids[i]: self.car_ips[i] for i in range(len(self.vehicle_ids))},
                car_bias={self.vehicle_ids[i]: self.car_bias[i] for i in range(len(self.vehicle_ids))},
                car_port=self.car_port,
                camera_rotation=self.camera_rotation
            )
            
            messagebox.showinfo("成功", "车辆配置已应用")
            self.update_result_text("车辆配置已更新")
            
        except Exception as e:
            messagebox.showerror("错误", f"配置应用失败: {str(e)}")

    def start_first_detection(self):
        """开始首次检测"""
        if self.first_detection_done:
            messagebox.showinfo("提示", "首次检测已完成")
            return
        
        try:
            camera = Camera(device_index=0)
    
            # 列出所有设备
            devices = camera.list_devices()
    
            if camera.connect():
            # 拍摄单张图片
                camera.capture_rotated_image("./captures/test_img.jpg", angle=self.camera_rotation)
            
            self.detect_latest_image()
            self.first_detection_done = True
            self.first_detect_btn.config(text="重新检测")
            self.detection_status.config(text="首次检测已完成", foreground="green")
            self.update_result_text("首次检测完成，已识别车辆ID")
            
        except Exception as e:
            messagebox.showerror("错误", f"首次检测失败: {str(e)}")
            self.detection_status.config(text="首次检测失败", foreground="red")
        finally:
        # 断开连接
            camera.disconnect()

    # 5. 修改 detect_latest_image 方法
    def detect_latest_image(self):
        """检测最新的图片"""
        captures_dir = "./captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
            return
        
        # 获取目录中所有图片文件
        image_files = [f for f in os.listdir(captures_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            messagebox.showwarning("警告", "captures目录中没有找到图片文件")
            return
        
        # 按修改时间排序，获取最新的图片
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(captures_dir, x)))
        latest_image = os.path.join(captures_dir, image_files[-1])
        self.background = mpimg.imread(latest_image)

        # 使用YOLO进行检测
        result = predict.detect_objects(latest_image)
        result_json = predict.save_detection_results(result, save_dir='./detection_results')
        
        if result_json:
            self.load_json_file(result_json)
            
            # 更新UI
            self.update_data_display()
            self.update_visualization()
            
            # 记录日志
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"{timestamp} 检测到最新图片: {latest_image}\n"
            log_msg += f"车辆: {len(self.grid_vehicles)}, 障碍物: {len(self.grid_obstacles)}, 目的地: {len(self.grid_destinations)}\n"
            log_msg += f"检测到的车辆ID: {list(range(len(self.grid_vehicles)))}\n"
            log_msg += f"配置的Vehicle IDs: {self.vehicle_ids}\n"
            self.result_text.insert(tk.END, log_msg + "\n")
            self.result_text.see(tk.END)
    
    def start_mission(self):
        """启动任务"""
        if not self.grid_vehicles or not self.grid_destinations:
            messagebox.showwarning("警告", "请先导入数据")
            return
        
        command = self.command_entry.get(1.0, tk.END).strip()
        if not command:
            messagebox.showwarning("警告", "请输入命令")
            return
        
        try:
            if self.vehicle_system.start_mission(command):
                self.stop_mission_btn.config(state=tk.NORMAL)
                self.update_result_text(f"任务启动成功: {command}")

                # 启动一个线程来监听路径结果并更新可视化
                threading.Thread(target=self.monitor_path_results, daemon=True).start()
            else:
                self.update_result_text("任务启动失败")
        except Exception as e:
            messagebox.showerror("错误", f"启动任务失败: {str(e)}")

    def stop_mission(self):
        """停止任务"""
        try:
            self.vehicle_system.cleanup()
            self.stop_mission_btn.config(state=tk.DISABLED)
            self.update_result_text("任务已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止任务失败: {str(e)}")
    
    def monitor_path_results(self):
        """监听系统的路径规划结果并更新可视化"""
        while self.vehicle_system.running:
            try:
                # 检查系统是否有路径结果
                if hasattr(self.vehicle_system, 'grid_path_results') and self.vehicle_system.grid_path_results:
                    # 在主线程中更新可视化
                    self.root.after(0, self.update_path_visualization, self.vehicle_system.path_results, self.vehicle_system.grid_path_results)
                    break
            except Exception as e:
                print(f"监听路径结果出错: {e}")
            
            time.sleep(0.5)  # 每0.5秒检查一次

    def update_path_visualization(self, path_results, grid_path_results):
        """更新路径可视化"""
        try:
            # 清除之前的绘图
            self.ax.clear()
            self.detect_latest_image()
            self.update_visualization()
            
            # 绘制路径
            if grid_path_results:
                self.draw_path_on_ax(grid_path_results)
                result_text = "路径规划完成：\n" + "\n".join(str(res) for res in path_results)
                self.result_text.insert(tk.END, result_text + "\n")
            
            for vid in self.vehicle_ids:
                traj = self.vehicle_system.get_actual_trajectory(vid)
                if not traj:
                    continue

                # 映射世界坐标 → 图像 → 网格（假设有 mapper）
                if hasattr(self.vehicle_system, "mapper") and self.vehicle_system.mapper.is_initialized:
                    img_points = self.vehicle_system.mapper.batch_map_to_real_coords(traj)
                    grid_points = batch_convert_to_image_coordinates(img_points)

                    x_vals = [p[0] for p in grid_points]
                    y_vals = [p[1] for p in grid_points]

                    self.ax.plot(x_vals, y_vals, color='blue', alpha=0.8, linewidth=2, label=f'车辆{vid}实际轨迹')

            self.canvas.draw()
            
        except Exception as e:
            self.update_result_text(f"可视化更新失败: {str(e)}") 

    def load_json_file(self, file_path=None):
        """加载JSON文件"""
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="选择JSON文件",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not file_path:
                return
        
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
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
            if hasattr(self, 'file_status'):
                self.file_status.config(text="文件导入失败", foreground="red")
    
    def validate_json_data(self, data):
        """验证JSON数据格式"""
        required_keys = ["all_vehicles", "obstacle", "destination"]
        return all(key in data for key in required_keys)
    
    def update_data_display(self):
        """更新数据显示"""
        self.data_text.delete(1.0, tk.END)
        
        info = f"车辆数量: {len(self.grid_vehicles)}\n"
        info += f"障碍物数量: {len(self.grid_obstacles)}\n"
        info += f"目的地数量: {len(self.grid_destinations)}\n\n"
        
        info += "车辆位置:\n"
        for i, vehicle in enumerate(self.grid_vehicles):
            info += f"  车辆{i}: {vehicle}\n"
        
        info += "\n障碍物位置:\n"
        for i, obstacle in enumerate(self.grid_obstacles):
            info += f"  障碍物{i}: {obstacle}\n"
        
        info += "\n目的地位置:\n"
        for i, dest in enumerate(self.grid_destinations):
            info += f"  目的地{i}: {dest}\n"
        
        self.data_text.insert(tk.END, info)
    
    def update_visualization(self):
        """更新可视化图形"""
        self.ax.clear()
        # 绘制背景图片（如果有）
        if hasattr(self, 'background') and self.background is not None:
            self.ax.imshow(self.background, extent=[0, 144, 0, 108], alpha=0.7, zorder=0)
            
        if not self.grid_vehicles and not self.grid_obstacles and not self.grid_destinations:
            self.ax.text(0.5, 0.5, "请导入JSON文件", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # 绘制车辆（蓝色矩形）
        for i, vehicle in enumerate(self.grid_vehicles):
            if len(vehicle) >= 4:
                x_min = min(point[0] for point in vehicle)
                y_min = min(point[1] for point in vehicle)
                width = max(point[0] for point in vehicle) - x_min
                height = max(point[1] for point in vehicle) - y_min
                
                rect = Rectangle((x_min, y_min), width, height, 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
                self.ax.add_patch(rect)
                
                # 添加车辆标签
                center_x = x_min + width / 2
                center_y = y_min + height / 2
                self.ax.text(center_x, center_y, f'V{i}', 
                           ha='center', va='center', fontweight='bold')
        
        # 绘制障碍物（红色矩形）
        for i, obstacle in enumerate(self.grid_obstacles):
            if len(obstacle) >= 4:
                x_min = min(point[0] for point in obstacle)
                y_min = min(point[1] for point in obstacle)
                width = max(point[0] for point in obstacle) - x_min
                height = max(point[1] for point in obstacle) - y_min
                
                rect = Rectangle((x_min, y_min), width, height, 
                               facecolor='lightcoral', edgecolor='red', linewidth=2)
                self.ax.add_patch(rect)
                
                # 添加障碍物标签
                center_x = x_min + width / 2
                center_y = y_min + height / 2
                self.ax.text(center_x, center_y, f'O{i}', 
                           ha='center', va='center', fontweight='bold')
        
        # 绘制目的地（绿色矩形）
        for i, dest in enumerate(self.grid_destinations):
            if len(dest) >= 4:
                x_min = min(point[0] for point in dest)
                y_min = min(point[1] for point in dest)
                width = max(point[0] for point in dest) - x_min
                height = max(point[1] for point in dest) - y_min
                
                rect = Rectangle((x_min, y_min), width, height, 
                               facecolor='lightgreen', edgecolor='green', linewidth=2)
                self.ax.add_patch(rect)
                
                # 添加目的地标签
                center_x = x_min + width / 2
                center_y = y_min + height / 2
                self.ax.text(center_x, center_y, f'D{i}', 
                           ha='center', va='center', fontweight='bold')
        
        # 设置图形属性
        self.ax.set_xlim(0, 144)
        self.ax.set_ylim(0, 108)
        self.ax.set_xlabel('X坐标')
        self.ax.set_ylabel('Y坐标')
        self.ax.set_title('车辆路径规划可视化')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue', label='车辆'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='red', label='障碍物'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='green', label='目的地')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        self.canvas.draw()

    def draw_path_on_ax(self, path_list):
        """将路径绘制到已有的 self.ax 上
        
        Args:
            path_list: 路径列表，每个路径是点坐标列表或包含路径数据的字典
        """
        if not path_list:
            return
        
        processed_paths = []
        for path_dict in path_list:      
            if isinstance(path_dict, dict):
                # 格式：{0: [...], 1: [...]}
                for vid, path in path_dict.items():
                    if isinstance(path, list):
                        processed_paths.append((vid, path))
        
        # 开始绘图
        for vehicle_index, path in processed_paths:
            if not path:
                continue

            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]

            # 绘制路径线
            self.ax.plot(path_x, path_y, color='orange', linewidth=2, linestyle='--',
                        label='规划路径', marker='o', markersize=4, alpha=0.8)

            # 标注路径点序号
            for i, (x, y) in enumerate(path):
                self.ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                                fontsize=8, color='red', weight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            # 起点和终点标记
            start_point = path[0]
            end_point = path[-1]
            self.ax.plot(start_point[0], start_point[1], 'go', markersize=12, label='起点')
            self.ax.plot(end_point[0], end_point[1], 'ro', markersize=12, label='终点')

            # 添加车辆编号标签
            if vehicle_index is not None:
                self.ax.text(start_point[0], start_point[1] + 1.0, f'车辆{vehicle_index}',
                            fontsize=9, color='black', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6))

    def execute_planning(self):
        """执行路径规划"""
        if not self.grid_vehicles or not self.grid_destinations:
            messagebox.showwarning("警告", "请先导入包含车辆和目的地数据的JSON文件")
            return
        
        command = self.command_entry.get(1.0, tk.END).strip()
        if not command:
            messagebox.showwarning("警告", "请输入命令")
            return
        
        self.start_mission()

    def setup_system_callbacks(self):
        """设置系统回调函数"""
        def status_callback(state, message):
            # 在结果显示区域更新状态
            self.root.after(0, lambda: self.update_result_text(f"状态: {state} - {message}"))
        
        def error_callback(error_msg):
            self.root.after(0, lambda: self.update_result_text(f"错误: {error_msg}"))
        
        def progress_callback(progress, message):
            self.root.after(0, lambda: self.update_result_text(f"进度: {progress}% - {message}"))
        
        self.vehicle_system.set_callbacks(status_callback, error_callback, progress_callback)

    def update_result_text(self, message):
        """线程安全的结果文本更新"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.result_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.result_text.see(tk.END)

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'vehicle_system'):
            self.vehicle_system.cleanup()

def main():
    root = tk.Tk()
    app = VehiclePlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# export ROS_MASTER_URI=http://192.168.1.214:11311
# roslaunch vrpn_client_ros sample.launch server:=192.168.0.2