import test
import algorithms
import draw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import numpy as np

class VehiclePlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆路径规划系统")
        self.root.geometry("1200x800")
        
        # 数据存储
        self.vehicles = []
        self.obstacles = []
        self.destinations = []
        
        # 创建界面
        self.create_widgets()
        
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
        
        self.file_status = ttk.Label(file_frame, text="未导入文件", 
                                   foreground="red")
        self.file_status.pack(pady=(5, 0))
        
        # 数据显示区域
        data_frame = ttk.LabelFrame(parent, text="数据信息", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.data_text = scrolledtext.ScrolledText(data_frame, height=8, width=35)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
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
        
    def load_json_file(self):
        """加载JSON文件"""
        file_path = filedialog.askopenfilename(
            title="选择JSON文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证数据格式
                if self.validate_json_data(data):
                    self.vehicles = data.get("all_vehicles", [])
                    self.obstacles = data.get("obstacle", [])
                    self.destinations = data.get("destination", [])
                    
                    self.file_status.config(text="文件导入成功", foreground="green")
                    self.update_data_display()
                    self.update_visualization()
                else:
                    raise ValueError("JSON格式不正确")
                    
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {str(e)}")
                self.file_status.config(text="文件导入失败", foreground="red")
    
    def validate_json_data(self, data):
        """验证JSON数据格式"""
        required_keys = ["all_vehicles", "obstacle", "destination"]
        return all(key in data for key in required_keys)
    
    def update_data_display(self):
        """更新数据显示"""
        self.data_text.delete(1.0, tk.END)
        
        info = f"车辆数量: {len(self.vehicles)}\n"
        info += f"障碍物数量: {len(self.obstacles)}\n"
        info += f"目的地数量: {len(self.destinations)}\n\n"
        
        info += "车辆位置:\n"
        for i, vehicle in enumerate(self.vehicles):
            info += f"  车辆{i}: {vehicle}\n"
        
        info += "\n障碍物位置:\n"
        for i, obstacle in enumerate(self.obstacles):
            info += f"  障碍物{i}: {obstacle}\n"
        
        info += "\n目的地位置:\n"
        for i, dest in enumerate(self.destinations):
            info += f"  目的地{i}: {dest}\n"
        
        self.data_text.insert(tk.END, info)
    
    def update_visualization(self):
        """更新可视化图形"""
        self.ax.clear()
        
        if not self.vehicles and not self.obstacles and not self.destinations:
            self.ax.text(0.5, 0.5, "请导入JSON文件", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # 绘制车辆（蓝色矩形）
        for i, vehicle in enumerate(self.vehicles):
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
        for i, obstacle in enumerate(self.obstacles):
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
        for i, dest in enumerate(self.destinations):
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
        """将路径绘制到已有的 self.ax 上"""
        for path in path_list:
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

    def execute_planning(self):
        """执行路径规划"""
        if not self.vehicles or not self.destinations:
            messagebox.showwarning("警告", "请先导入包含车辆和目的地数据的JSON文件")
            return
        
        command = self.command_entry.get(1.0, tk.END).strip()
        if not command:
            messagebox.showwarning("警告", "请输入命令")
            return
        
        try:
            # 创建路径规划对象
            obj = algorithms.PathPlanner(self.vehicles, self.obstacles, self.destinations)
            
            # 调用LLM
            function_list = test.call_LLM(self.vehicles, self.destinations, command)
            
            if function_list is None:
                result = "Error: LLM failed!"
            else:
                path = test.Interpret_function_list(function_list, obj)
                result = "结果为：\n" + str(path)

            self.update_visualization()
            self.draw_path_on_ax(path)
            self.canvas.draw()
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
            
        except Exception as e:
            messagebox.showerror("错误", f"执行失败: {str(e)}")

def main():
    root = tk.Tk()
    app = VehiclePlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()