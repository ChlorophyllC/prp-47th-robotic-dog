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

class VehiclePlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆路径规划系统")
        self.root.geometry("1200x800")
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
        
        # 自动检测控制区域
        detect_frame = ttk.LabelFrame(parent, text="自动检测控制", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.auto_detect_btn = ttk.Button(detect_frame, text="开始自动检测", 
                                        command=self.toggle_auto_detection)
        self.auto_detect_btn.pack(fill=tk.X)
        
        self.file_status = ttk.Label(file_frame, text="未导入文件", 
                                   foreground="red")
        self.file_status.pack(pady=(5, 0))
        
        # 数据显示区域
        data_frame = ttk.LabelFrame(parent, text="数据信息", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.data_text = scrolledtext.ScrolledText(data_frame, height=8, width=35)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        system_frame = ttk.LabelFrame(parent, text="系统控制", padding=10)
        system_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_mission_btn = ttk.Button(system_frame, text="启动任务", 
                                        command=self.start_mission)
        self.start_mission_btn.pack(fill=tk.X, pady=(0, 5))

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
                self.start_mission_btn.config(state=tk.DISABLED)
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
            self.start_mission_btn.config(state=tk.NORMAL)
            self.stop_mission_btn.config(state=tk.DISABLED)
            self.update_result_text("任务已停止")
        except Exception as e:
            messagebox.showerror("错误", f"停止任务失败: {str(e)}")
    
    def monitor_path_results(self):
        """监听系统的路径规划结果并更新可视化"""
        while self.vehicle_system.running:
            try:
                # 检查系统是否有路径结果
                if hasattr(self.vehicle_system, 'path_results') and self.vehicle_system.path_results:
                    # 在主线程中更新可视化
                    self.root.after(0, self.update_path_visualization, self.vehicle_system.path_results)
                    break
            except Exception as e:
                print(f"监听路径结果出错: {e}")
            
            time.sleep(0.5)  # 每0.5秒检查一次

    def update_path_visualization(self, path_results):
        """更新路径可视化"""
        try:
            # 清除之前的绘图
            self.ax.clear()
            self.update_visualization()
            
            # 绘制路径
            if path_results:
                self.draw_path_on_ax(path_results)
                result_text = "路径规划完成：\n" + "\n".join(str(res) for res in path_results)
                self.result_text.insert(tk.END, result_text + "\n")
            
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
    
    def start_auto_detection(self):
        """启动自动检测线程"""
        self.auto_detect_running = True
        self.detection_thread = threading.Thread(target=self.auto_detect_loop, daemon=True)
        self.detection_thread.start()
    
    def stop_auto_detection(self):
        """停止自动检测"""
        self.auto_detect_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join()
    
    def toggle_auto_detection(self):
        """切换自动检测状态"""
        if self.auto_detect_running:
            self.stop_auto_detection()
            self.auto_detect_btn.config(text="开始自动检测")
        else:
            self.start_auto_detection()
            self.auto_detect_btn.config(text="停止自动检测")
    
    def auto_detect_loop(self):
        """自动检测循环"""
        while self.auto_detect_running:
            try:
                self.detect_latest_image()
            except Exception as e:
                print(f"自动检测出错: {e}")
            
            # 每隔5秒检测一次
            for _ in range(5):
                if not self.auto_detect_running:
                    return
                time.sleep(1)
    
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
            return
        
        # 按修改时间排序，获取最新的图片
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(captures_dir, x)))
        latest_image = os.path.join(captures_dir, image_files[-1])
        self.background = mpimg.imread(latest_image)

        # 使用YOLO进行检测
        result = predict.detect_objects(latest_image)
        result_json = predict.save_detection_results(result, save_dir='./detection_results')
        self.load_json_file(result_json)

        # 更新数据
        if result_json:
            
            # 更新UI
            self.root.after(0, self.update_data_display)
            self.root.after(0, self.update_visualization)
            
            # 记录日志
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"{timestamp} 检测到最新图片: {latest_image}\n"
            log_msg += f"车辆: {len(self.grid_vehicles)}, 障碍物: {len(self.grid_obstacles)}, 目的地: {len(self.grid_destinations)}\n"
            self.result_text.insert(tk.END, log_msg + "\n")
            self.result_text.see(tk.END)
        
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
            self.ax.imshow(self.background, extent=[0, 72, 0, 54], alpha=0.7, zorder=0)
            
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
        self.ax.set_xlim(0, 72)
        self.ax.set_ylim(0, 54)
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
        """析构函数，确保线程停止"""
        self.stop_auto_detection()
        if hasattr(self, 'vehicle_system'):
            self.vehicle_system.cleanup()

def main():
    root = tk.Tk()
    app = VehiclePlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# export ROS_MASTER_URI=http://192.168.1.214:11311
# roslaunch vrpn_client_ros sample.launch server:=192.168.1.100