import tkinter as tk
from tkinter import messagebox, filedialog
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import tkinter as tk
from tkinter import messagebox, filedialog
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class JSONGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JSON 可视化生成器")
        
        # 初始化数据结构
        self.data = {
            "all_vehicles": [],
            "obstacle": [],
            "destination": []
        }
        
        # 当前编辑模式
        self.current_mode = "vehicle"  # vehicle, obstacle, destination
        self.current_points = []
        
        # 创建 GUI 布局
        self.create_widgets()
        
        # 设置网格和绘图
        self.setup_plot()
    
    def create_widgets(self):
        # 左侧控制面板
        control_frame = tk.Frame(self.root, width=200, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # 模式选择
        mode_label = tk.Label(control_frame, text="选择编辑模式:")
        mode_label.pack(pady=(10, 5))
        
        self.mode_var = tk.StringVar(value=self.current_mode)
        
        vehicle_btn = tk.Radiobutton(control_frame, text="车辆", variable=self.mode_var, 
                                    value="vehicle", command=self.change_mode)
        obstacle_btn = tk.Radiobutton(control_frame, text="障碍物", variable=self.mode_var, 
                                      value="obstacle", command=self.change_mode)
        destination_btn = tk.Radiobutton(control_frame, text="目的地", variable=self.mode_var, 
                                         value="destination", command=self.change_mode)
        
        vehicle_btn.pack(anchor=tk.W)
        obstacle_btn.pack(anchor=tk.W)
        destination_btn.pack(anchor=tk.W)
        
        # 操作按钮
        tk.Button(control_frame, text="删除最后一个点", command=self.remove_last_point).pack(pady=10)
        tk.Button(control_frame, text="清空当前模式", command=self.clear_current_mode).pack(pady=5)
        tk.Button(control_frame, text="清空所有", command=self.clear_all).pack(pady=5)
        
        # 保存按钮
        tk.Button(control_frame, text="保存为JSON", command=self.save_json).pack(pady=(20, 5))
        
        # 右侧绘图区域
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # 创建 Matplotlib 图形
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # 将图形嵌入 Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
        # 绑定点击事件
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def setup_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 20)
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.set_xticks(np.arange(0, 21, 1))
        self.ax.set_yticks(np.arange(0, 21, 1))
        self.ax.set_title("点击添加点 (4个点自动生成四边形)")
        
        # 绘制已有元素
        self.draw_existing_elements()
        
        # 绘制当前正在编辑的多边形
        if self.current_points:
            self.draw_current_polygon()
        
        self.canvas.draw()
    
    def draw_existing_elements(self):
        # 绘制车辆
        for i, vehicle in enumerate(self.data["all_vehicles"]):
            polygon = patches.Polygon(vehicle, closed=True, fill=True, 
                                     color='blue', alpha=0.5, label=f"Vehicle {i}")
            self.ax.add_patch(polygon)
            # 添加标签
            centroid = np.mean(vehicle, axis=0)
            self.ax.text(centroid[0], centroid[1], f"V{i}", 
                         ha='center', va='center', color='white')
        
        # 绘制障碍物
        for i, obstacle in enumerate(self.data["obstacle"]):
            polygon = patches.Polygon(obstacle, closed=True, fill=True, 
                                     color='red', alpha=0.5, label=f"Obstacle {i}")
            self.ax.add_patch(polygon)
            # 添加标签
            centroid = np.mean(obstacle, axis=0)
            self.ax.text(centroid[0], centroid[1], f"O{i}", 
                         ha='center', va='center', color='white')
        
        # 绘制目的地
        for i, destination in enumerate(self.data["destination"]):
            polygon = patches.Polygon(destination, closed=True, fill=True, 
                                     color='green', alpha=0.5, label=f"Destination {i}")
            self.ax.add_patch(polygon)
            # 添加标签
            centroid = np.mean(destination, axis=0)
            self.ax.text(centroid[0], centroid[1], f"D{i}", 
                         ha='center', va='center', color='white')
    
    def draw_current_polygon(self):
        if len(self.current_points) > 1:
            # 绘制线条
            x, y = zip(*self.current_points)
            self.ax.plot(x, y, 'b--', marker='o', markersize=5)
        
        # 绘制点
        for point in self.current_points:
            self.ax.plot(point[0], point[1], 'ro')
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # 如果已经有4个点了，提示用户
        if len(self.current_points) >= 4:
            messagebox.showinfo("提示", "已经有4个点了，请先删除多余的点或切换模式!")
            return
        
        # 获取点击坐标
        x, y = int(round(event.xdata)), int(round(event.ydata))
        
        # 添加到当前点列表
        self.current_points.append((x, y))
        
        # 如果已经有4个点，自动完成四边形
        if len(self.current_points) == 4:
            self.auto_finish_quadrilateral()
        else:
            # 重绘
            self.setup_plot()
    
    def auto_finish_quadrilateral(self):
        """自动完成四边形生成"""
        # 根据当前模式添加到数据结构
        if self.current_mode == "vehicle":
            self.data["all_vehicles"].append(self.current_points)
        elif self.current_mode == "obstacle":
            self.data["obstacle"].append(self.current_points)  
        elif self.current_mode == "destination":
            self.data["destination"].append(self.current_points)
        
        # 重置当前点
        self.current_points = []
        self.setup_plot()
    
    def change_mode(self):
        self.current_mode = self.mode_var.get()
        self.current_points = []
        self.setup_plot()
    
    def finish_polygon(self):
        if len(self.current_points) != 4:
            messagebox.showwarning("警告", "必须有且仅有4个点来创建一个四边形!")
            return
        
        # 根据当前模式添加到数据结构
        if self.current_mode == "vehicle":
            self.data["all_vehicles"].append(self.current_points)
        elif self.current_mode == "obstacle":
            self.data["obstacle"].append(self.current_points)
        elif self.current_mode == "destination":
            if self.data["destination"]:
                messagebox.showwarning("警告", "只能有一个目的地!")
                return
            self.data["destination"] = self.current_points
        
        # 重置当前点
        self.current_points = []
        self.setup_plot()
    
    def remove_last_point(self):
        if self.current_points:
            self.current_points.pop()
            self.setup_plot()
    
    def clear_current_mode(self):
        if self.current_mode == "vehicle":
            self.data["all_vehicles"] = []
        elif self.current_mode == "obstacle":
            self.data["obstacle"] = []
        elif self.current_mode == "destination":
            self.data["destination"] = []
        
        self.current_points = []
        self.setup_plot()
    
    def clear_all(self):
        self.data = {
            "all_vehicles": [],
            "obstacle": [],
            "destination": []
        }
        self.current_points = []
        self.setup_plot()
    
    def format_polygon_list(self, polygon_list, indent_level=1):
        """格式化四边形列表，使用紧凑格式"""
        if not polygon_list:
            return "[]"
        
        base_indent = "    " * indent_level
        items = []
        
        for quadrilateral in polygon_list:
            # 将每个四边形格式化为一行，点用元组表示
            points_str = ", ".join([f"[{x}, {y}]" for x, y in quadrilateral])
            items.append(f"[{points_str}]")
        
        if len(items) == 1:
            return f"[\n{base_indent}{items[0]}\n{base_indent[:-4]}]"
        else:
            formatted_items = f",\n{base_indent}".join(items)
            return f"[\n{base_indent}{formatted_items}\n{base_indent[:-4]}]"
    
    def format_single_polygon(self, quadrilateral, indent_level=1):
        """格式化单个四边形"""
        if not quadrilateral:
            return "[]"
        
        base_indent = "    " * indent_level
        points_str = ", ".join([f"[{x}, {y}]" for x, y in quadrilateral])
        return f"[\n{base_indent}{points_str}\n{base_indent[:-4]}]"
    
    def save_json(self):
        # 检查是否有未完成的四边形
        if self.current_points:
            response = messagebox.askyesno("保存", f"当前有{len(self.current_points)}个点（需要4个点），是否清除它们并保存?")
            if response:
                self.current_points = []
        
        # 获取保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 手动构建JSON字符串以实现紧凑格式
                json_content = "{\n"
                json_content += f'    "all_vehicles": {self.format_polygon_list(self.data["all_vehicles"])},\n\n'
                json_content += f'    "obstacle": {self.format_polygon_list(self.data["obstacle"])},\n\n'
                json_content += f'    "destination": {self.format_polygon_list(self.data["destination"])}\n'
                json_content += "}"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                
                messagebox.showinfo("成功", f"JSON文件已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = JSONGeneratorApp(root)
    root.mainloop()


def JSONGenerator():
    root = tk.Tk()
    app = JSONGeneratorApp(root)
    root.mainloop()

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        messagebox.showerror("错误", f"加载JSON文件时出错:\n{str(e)}")
        return None

data = load_json("sim.json")  # 替换为实际的JSON文件路径
print(data)