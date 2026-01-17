import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
from typing import List, Tuple
import threading
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')

class TrajectoryGUI:
    def __init__(self, root):
        self.root = root
        matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'
        self.root.title("轨迹可视化工具")
        self.root.geometry("1200x800")
        
        # 数据存储
        self.log_files = []
        self.trajectory_data = []
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        
        # 创建界面
        self.create_widgets()
        
        # 设置中文字体
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # 停止动画（如有）
        if hasattr(self, 'animation_active'):
            self.animation_active = False
        # 退出主循环并销毁窗口
        try:
            self.root.quit()
        except Exception:
            pass
        self.root.destroy()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(control_frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="选择日志文件", command=self.select_files).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(file_frame, text="清空选择", command=self.clear_files).pack(fill=tk.X, pady=(0, 5))
        
        # 文件列表
        self.file_listbox = tk.Listbox(file_frame, height=8)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # 绘图选项
        plot_frame = ttk.LabelFrame(control_frame, text="绘图选项", padding=10)
        plot_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 标签间隔
        ttk.Label(plot_frame, text="标签间隔(mm):").pack(anchor=tk.W)
        self.label_interval = tk.StringVar(value="100")
        ttk.Entry(plot_frame, textvariable=self.label_interval, width=10).pack(fill=tk.X, pady=(0, 5))
        
        # 显示选项
        self.show_actual = tk.BooleanVar(value=True)
        self.show_target = tk.BooleanVar(value=True)
        self.show_error = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(plot_frame, text="显示实际路径", variable=self.show_actual).pack(anchor=tk.W)
        ttk.Checkbutton(plot_frame, text="显示理想路径", variable=self.show_target).pack(anchor=tk.W)
        ttk.Checkbutton(plot_frame, text="显示误差曲线", variable=self.show_error).pack(anchor=tk.W)
        
        # 绘图按钮
        ttk.Button(plot_frame, text="绘制轨迹", command=self.plot_trajectories).pack(fill=tk.X, pady=(10, 0))
        
        # 动画选项
        animation_frame = ttk.LabelFrame(control_frame, text="动画选项", padding=10)
        animation_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 动画速度
        ttk.Label(animation_frame, text="动画间隔(ms):").pack(anchor=tk.W)
        self.animation_interval = tk.StringVar(value="50")
        ttk.Entry(animation_frame, textvariable=self.animation_interval, width=10).pack(fill=tk.X, pady=(0, 5))
        
        # GIF选项
        ttk.Label(animation_frame, text="GIF帧率(fps):").pack(anchor=tk.W)
        self.gif_fps = tk.StringVar(value="20")
        ttk.Entry(animation_frame, textvariable=self.gif_fps, width=10).pack(fill=tk.X, pady=(0, 5))
        
        # 动画按钮
        ttk.Button(animation_frame, text="播放动画", command=self.start_animation).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(animation_frame, text="生成GIF", command=self.generate_gif).pack(fill=tk.X, pady=(5, 0))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(pady=(5, 0))
        
        # 右侧绘图区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        
    def select_files(self):
        """选择日志文件"""
        files = filedialog.askopenfilenames(
            title="选择日志文件",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.log_files:
                self.log_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        self.status_label.config(text=f"已选择 {len(self.log_files)} 个文件")
    
    def clear_files(self):
        """清空文件选择"""
        self.log_files.clear()
        self.trajectory_data.clear()
        self.file_listbox.delete(0, tk.END)
        self.status_label.config(text="文件列表已清空")
    
    def point_to_segment_distance(self, p, a, b):
        """计算点p到线段ab的最小距离"""
        ap = p - a
        ab = b - a
        ab_norm_sq = np.dot(ab, ab)
        if ab_norm_sq == 0:
            return np.linalg.norm(ap)
        t = np.clip(np.dot(ap, ab) / ab_norm_sq, 0, 1)
        projection = a + t * ab
        return np.linalg.norm(p - projection)
    
    def load_trajectory_data(self, log_file_path):
        """加载轨迹数据"""
        try:
            data = np.genfromtxt(log_file_path, delimiter=',', skip_header=1,
                               names=['time', 'x', 'y', 'heading', 'target_x', 'target_y', 'target_idx'])
            
            actual_pos = np.column_stack((data['x'], data['y']))
            target_pos = np.column_stack((data['target_x'], data['target_y']))
            
            # 计算误差
            errors = []
            for p in actual_pos:
                min_dist = float('inf')
                for i in range(len(target_pos) - 1):
                    a = target_pos[i]
                    b = target_pos[i + 1]
                    dist = self.point_to_segment_distance(p, a, b)
                    if dist < min_dist:
                        min_dist = dist
                errors.append(min_dist)
            
            return {
                'time': data['time'],
                'actual_pos': actual_pos,
                'target_pos': target_pos,
                'errors': np.array(errors),
                'filename': os.path.basename(log_file_path)
            }
        except Exception as e:
            messagebox.showerror("错误", f"加载文件 {log_file_path} 时出错: {str(e)}")
            return None
    
    def plot_trajectories(self):
        """绘制多条轨迹"""
        if not self.log_files:
            messagebox.showwarning("警告", "请先选择日志文件")
            return
        
        self.status_label.config(text="正在加载数据...")
        self.progress_var.set(0)
        
        # 清空之前的数据
        self.trajectory_data.clear()
        
        # 加载所有轨迹数据
        for i, log_file in enumerate(self.log_files):
            data = self.load_trajectory_data(log_file)
            if data:
                self.trajectory_data.append(data)
            
            # 更新进度
            progress = (i + 1) / len(self.log_files) * 50
            self.progress_var.set(progress)
            self.root.update_idletasks()
        
        if not self.trajectory_data:
            messagebox.showerror("错误", "没有成功加载任何轨迹数据")
            return
        
        # 清空画布
        self.ax1.clear()
        self.ax2.clear()
        
        # 绘制轨迹
        label_interval = float(self.label_interval.get())
        
        for i, data in enumerate(self.trajectory_data):
            color = self.colors[i % len(self.colors)]
            filename = data['filename']
            
            # 绘制实际轨迹
            if self.show_actual.get():
                self.ax1.plot(data['actual_pos'][:, 0], data['actual_pos'][:, 1], 
                            color=color, linestyle='-', label=f'{filename} - 实际路径')
                
                # 标记起点和终点
                self.ax1.scatter(data['actual_pos'][0, 0], data['actual_pos'][0, 1], 
                               c=color, s=100, marker='o', alpha=0.7)
                self.ax1.scatter(data['actual_pos'][-1, 0], data['actual_pos'][-1, 1], 
                               c=color, s=100, marker='x', alpha=0.7)
            
            # 绘制理想轨迹
            if self.show_target.get():
                self.ax1.plot(data['target_pos'][:, 0], data['target_pos'][:, 1], 
                            color=color, linestyle='--', alpha=0.7, label=f'{filename} - 理想路径')
            
            # 绘制误差曲线
            if self.show_error.get():
                time_rel = data['time'] - data['time'][0]
                self.ax2.plot(time_rel, data['errors'], color=color, label=f'{filename} - 误差')
        
        # 设置图表属性
        self.ax1.set_title('多轨迹对比')
        self.ax1.set_xlabel('X坐标(mm)')
        self.ax1.set_ylabel('Y坐标(mm)')
        self.ax1.legend()
        self.ax1.grid(True)
        self.ax1.axis('equal')
        
        if self.show_error.get():
            self.ax2.set_title('路径误差对比')
            self.ax2.set_xlabel('时间 (s)')
            self.ax2.set_ylabel('误差 (mm)')
            self.ax2.legend()
            self.ax2.grid(True)
        
        self.canvas.draw()
        self.progress_var.set(100)
        self.status_label.config(text="绘制完成")
    
    def start_animation(self):
        """开始动画播放"""
        if not self.trajectory_data:
            messagebox.showwarning("警告", "请先绘制轨迹")
            return
        
        self.status_label.config(text="正在播放动画...")
        
        # 创建动画
        self.animation_active = True
        interval = int(self.animation_interval.get())
        
        def animate(frame):
            if not self.animation_active:
                return
            
            self.ax1.clear()
            
            for i, data in enumerate(self.trajectory_data):
                color = self.colors[i % len(self.colors)]
                filename = data['filename']
                
                # 计算当前帧应该显示的点数
                max_points = len(data['actual_pos'])
                current_points = min(frame * 2, max_points)  # 每帧显示2个点
                
                if current_points > 0:
                    # 绘制轨迹（从起点到当前点）
                    self.ax1.plot(data['actual_pos'][:current_points, 0], 
                                data['actual_pos'][:current_points, 1], 
                                color=color, linestyle='-', label=f'{filename}')
                    
                    # 绘制当前位置点
                    self.ax1.scatter(data['actual_pos'][current_points-1, 0], 
                                   data['actual_pos'][current_points-1, 1], 
                                   c=color, s=50, marker='o')
            
            self.ax1.set_title(f'轨迹动画 - 帧 {frame}')
            self.ax1.set_xlabel('X坐标(mm)')
            self.ax1.set_ylabel('Y坐标(mm)')
            self.ax1.legend()
            self.ax1.grid(True)
            self.ax1.axis('equal')
        
        # 计算总帧数
        max_frames = max(len(data['actual_pos']) for data in self.trajectory_data) // 2
        
        self.ani = animation.FuncAnimation(self.fig, animate, frames=max_frames, 
                                         interval=interval, repeat=True)
        self.canvas.draw()
        
        # 添加停止按钮（临时）
        def stop_animation():
            self.animation_active = False
            self.status_label.config(text="动画已停止")
        
        # 在界面上添加停止按钮（简化实现）
        messagebox.showinfo("动画控制", "动画已开始播放。关闭此对话框后，动画将继续播放。")
    
    def generate_gif(self):
        """生成GIF动画"""
        if not self.trajectory_data:
            messagebox.showwarning("警告", "请先绘制轨迹")
            return
        
        # 选择保存路径
        gif_path = filedialog.asksaveasfilename(
            title="保存GIF动画",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")]
        )
        
        if not gif_path:
            return
        
        def generate_gif_thread():
            try:
                self.status_label.config(text="正在生成GIF...")
                fps = int(self.gif_fps.get())
                
                # 创建新的图形用于GIF生成
                fig, ax = plt.subplots(figsize=(10, 8))
                
                frames = []
                max_frames = max(len(data['actual_pos']) for data in self.trajectory_data) // 2
                
                for frame in range(max_frames):
                    ax.clear()
                    
                    for i, data in enumerate(self.trajectory_data):
                        color = self.colors[i % len(self.colors)]
                        filename = data['filename']
                        
                        current_points = min(frame * 2, len(data['actual_pos']))
                        
                        if current_points > 0:
                            ax.plot(data['actual_pos'][:current_points, 0], 
                                   data['actual_pos'][:current_points, 1], 
                                   color=color, linestyle='-', label=f'{filename}')
                            
                            ax.scatter(data['actual_pos'][current_points-1, 0], 
                                     data['actual_pos'][current_points-1, 1], 
                                     c=color, s=50, marker='o')
                    
                    ax.set_title(f'轨迹动画 - 帧 {frame}')
                    ax.set_xlabel('X坐标(mm)')
                    ax.set_ylabel('Y坐标(mm)')
                    ax.legend()
                    ax.grid(True)
                    ax.axis('equal')
                    
                    # 保存帧
                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(Image.fromarray(img))
                    
                    # 更新进度
                    progress = (frame + 1) / max_frames * 100
                    self.progress_var.set(progress)
                    self.root.update_idletasks()
                
                # 保存GIF
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                             duration=1000//fps, loop=0)
                
                plt.close(fig)
                self.status_label.config(text=f"GIF已保存到: {gif_path}")
                messagebox.showinfo("成功", f"GIF动画已保存到:\n{gif_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"生成GIF时出错: {str(e)}")
                self.status_label.config(text="GIF生成失败")
            finally:
                self.progress_var.set(0)
        
        # 在后台线程中生成GIF
        thread = threading.Thread(target=generate_gif_thread)
        thread.daemon = True
        thread.start()

def main():
    root = tk.Tk()
    app = TrajectoryGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()