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
from predict import batch_convert_to_grid_coordinates
import cv2
import queue
import sys
import math
from typing import Optional
class VehiclePlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("车辆路径规划系统")
        self.root.geometry("1600x1200")
        matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'

        # 关闭标志必须尽早初始化：create_widgets() 过程中会触发 update_visualization()
        # 进而调用 _request_canvas_draw()，否则会出现 AttributeError。
        self._closing = False

        # 坐标映射器策略：是否在下一次任务启动时强制重建 mapper.pkl
        # 注意：必须在 create_widgets() 之前初始化（UI 会绑定该变量）
        self.force_rebuild_mapper_var = tk.BooleanVar(value=False)

        # 可视化开关：避障叠加/执行期背景刷新（必须在 create_widgets() 前初始化）
        self.show_avoidance_overlay_var = tk.BooleanVar(value=True)
        self.refresh_background_during_run_var = tk.BooleanVar(value=False)
        # 仅控制“虚线紫框（其他车的动态避让膨胀框）”显示
        self.show_dynamic_obstacle_boxes_var = tk.BooleanVar(value=True)
        
        # 车辆配置参数（先初始化，再创建系统；否则系统会用默认 rotation/ids，导致 pkl 判断与执行阶段不一致）
        self.vehicle_ids = [0, 1, 2]  # 默认值（与界面默认一致）
        self.car_ips = {0: "192.168.1.224", 1: "192.168.1.227", 2: "192.168.1.229"}  # 默认值，keys为vehicle_id
        self.car_bias = {0: 0, 1: 0, 2: 0}  # 默认值，keys为vehicle_id
        self.car_port = 12345  # 默认值
        self.camera_rotation = -6

        # 数据存储
        self.grid_vehicles = []
        self.grid_obstacles = []
        self.grid_destinations = []
        
        # 自动检测标志
        self.background = None

        # 背景显示分辨率（折中：清晰度 vs Matplotlib 重绘开销；保持 4:3）
        # 之前用 144x108 会太糊；720x540 又容易卡。
        self._bg_display_size = (720, 540)

        # 背景图 artist（用于低频刷新 set_data，不重建坐标系）
        self._bg_artist = None

        # 画布重绘节流：把多处 draw_idle 合并，避免 20Hz 多源触发导致持续重绘卡顿
        self._draw_after_id = None
        self._last_draw_ts = 0.0
        # 折中：比 20Hz 略低，但足够顺滑且明显减负（可按需改成 50/66/100）
        self._draw_min_interval_ms = 80

        # 高频日志（progress）去抖：避免 ScrolledText 频繁 insert/delete 造成卡顿
        self._last_progress_log_ts = 0.0
        self._last_progress_value = None

        # state 日志也做去抖：有些系统会高频重复推送同一状态
        self._last_state_log_ts = 0.0
        self._last_state_value = None

        # 避障叠加：缓存最近一次避障重规划结果（来自 system 事件）
        # {vehicle_id: {grid_path, ts, hit_point, grid_other_vehicle_obstacles, ...}}
        self._avoidance_state = {}
        # 避障 artist 缓存：避免频繁清空重画
        self._avoid_artists = {}
        self._avoid_hit_artists = {}
        self._avoid_dyn_obs_artists = []

        # 执行期背景刷新循环（默认关闭）
        self._bg_refresh_id = None
        self._bg_refresh_interval_ms = 50  # 20Hz

        # 车辆控制系统（把 GUI 默认配置传入，避免系统用默认值导致任务启动/映射判断混乱）
        self.vehicle_system = VehicleControlSystem(
            vehicle_ids=self.vehicle_ids,
            car_ips=self.car_ips,
            car_bias=self.car_bias,
            car_port=self.car_port,
            camera_rotation=self.camera_rotation,
        )
        self.setup_system_callbacks()

        # 创建界面
        self.create_widgets()
        self.color_palette = [
        ('#FF6B6B', '#FF8E8E'),  # 珊瑚红
        ('#4ECDC4', '#88E0D0'),  # 蓝绿色
        ('#8338EC', '#9D5BFF'),  # 紫色
        ('#3A86FF', '#6BA4FF'),  # 亮蓝色
        ('#FB5607', '#FF7B3D'),  # 橙红色
        ('#00BB9E', '#00D4B1'),  # 翡翠绿
        ('#FF006E', '#FF4D97'),  # 玫红色
        ('#A5DD9B', '#C1E8B7'),  # 薄荷绿
        ('#7E6B8F', '#9D8DAC')   # 薰衣草紫
        ]
        
        # 车辆配置参数（已在 system 初始化前设置）
        self.trajectories = {}  # 格式: {vehicle_id: [(x1,y1), (x2,y2), ...]}
        
        # 首次检测标志
        self.first_detection_done = False
        self.path_planned = False
        self._monitor_running = False
        
        # YOLO检测到真实小车的映射 {yolo_index: real_vehicle_id}
        self.yolo_to_real_mapping = {}
        
        # 线程管理
        self._detection_thread = None
        self._closing = False

        # 系统事件队列（事件驱动更新 UI）
        self._event_queue: "queue.Queue[dict]" = queue.Queue()
        self._event_poller_id = None
        self._system_cleaned = False

        # 实时可视化（ROS 轨迹叠加绘制）
        self._realtime_viz_id = None
        # 轨迹刷新比 UI 主循环更贵（含坐标映射/批量变换），默认 10Hz 更稳；如需可再调高
        self._realtime_viz_interval_ms = 100  # 10Hz

        # 轨迹绘图对象缓存：避免每次清空/重画导致卡顿
        # {vehicle_id: {"line": Line2D, "point": PathCollection, "text": Text, "start": PathCollection}}
        self._traj_artists = {}

        # 子窗口/资源跟踪（避免 Toplevel + Matplotlib 残留导致退出卡住）
        self._child_windows = set()
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 启动事件轮询（主线程消费队列）
        self._start_event_poller()

        # 启动实时轨迹刷新循环（不频繁刷新照片，只叠加绘制 ROS 轨迹）
        self._start_realtime_viz_loop()

        # 启动低频背景刷新循环（从 captures 读取最新图片；不强制抓拍）
        self._start_background_refresh_loop()

    def _start_background_refresh_loop(self):
        """任务执行期低频刷新背景图（可选开关）。

        说明：这里只做“从 captures 目录读取最新图片并更新 imshow.set_data()”。
        不主动抓拍，避免与检测线程/相机资源竞争。
        """

        def tick():
            if self._closing:
                return

            try:
                enabled = bool(self.refresh_background_during_run_var.get())
            except Exception:
                enabled = False

            running = False
            try:
                running = bool(getattr(self.vehicle_system, 'running', False))
            except Exception:
                running = False

            if enabled and running:
                try:
                    updated = self._refresh_background_artist_only()
                    if updated:
                        self._request_canvas_draw()
                except Exception:
                    pass

            self._bg_refresh_id = self.root.after(self._bg_refresh_interval_ms, tick)

        tick()

    def _refresh_background_artist_only(self) -> bool:
        """只刷新背景图 artist，不清空/重画其他叠加层。"""
        try:
            self.set_background()
        except Exception:
            return False

        if self.background is None:
            return False

        try:
            if self._bg_artist is not None:
                self._bg_artist.set_data(self.background)
                return True
        except Exception:
            self._bg_artist = None

        try:
            self._bg_artist = self.ax.imshow(self.background, extent=[0, 144, 0, 108], alpha=0.7, zorder=0)
            return True
        except Exception:
            return False

    def _clear_avoidance_artists(self) -> None:
        try:
            for vid, artists in list(self._avoid_artists.items()):
                for k in ("line", "points", "text"):
                    a = artists.get(k)
                    if a is None:
                        continue
                    try:
                        a.remove()
                    except Exception:
                        pass
            self._avoid_artists.clear()
        except Exception:
            pass

        try:
            for vid, a in list(self._avoid_hit_artists.items()):
                try:
                    a.remove()
                except Exception:
                    pass
            self._avoid_hit_artists.clear()
        except Exception:
            pass

        try:
            for a in list(self._avoid_dyn_obs_artists):
                try:
                    a.remove()
                except Exception:
                    pass
            self._avoid_dyn_obs_artists = []
        except Exception:
            pass

    def _apply_avoidance_overlay_update(self, event: dict) -> None:
        """将一次避障重规划结果叠加到当前坐标系（不清空画布）。"""
        try:
            if not bool(self.show_avoidance_overlay_var.get()):
                return
        except Exception:
            return

        vid = event.get('vehicle_id')
        try:
            vid = int(vid)
        except Exception:
            return

        grid_path = event.get('grid_path') or []
        if not grid_path or len(grid_path) < 2:
            return

        xs = [float(p[0]) for p in grid_path]
        ys = [float(p[1]) for p in grid_path]

        artists = self._avoid_artists.get(vid)
        if not artists:
            line = self.ax.plot(
                xs,
                ys,
                color='#FF00FF',
                linewidth=4.0,
                linestyle='-',
                alpha=0.9,
                zorder=12,
                label='避障重规划',
            )[0]
            points = self.ax.scatter(
                xs,
                ys,
                s=28,
                color='#FF66FF',
                alpha=0.75,
                zorder=13,
                edgecolors='white',
                linewidths=0.8,
            )

            last_x, last_y = xs[-1], ys[-1]
            text = self.ax.text(
                last_x,
                last_y + 2.0,
                f"避障路径 vehicle_{vid}",
                fontsize=11,
                color='white',
                fontweight='bold',
                ha='center',
                va='center',
                zorder=14,
                bbox=dict(facecolor='#AA00AA', alpha=0.65, boxstyle='round,pad=0.25', edgecolor='white'),
            )
            self._avoid_artists[vid] = {"line": line, "points": points, "text": text}
        else:
            line = artists.get('line')
            points = artists.get('points')
            text = artists.get('text')
            if line is not None:
                try:
                    line.set_data(xs, ys)
                except Exception:
                    pass
            if points is not None:
                try:
                    points.set_offsets(list(zip(xs, ys)))
                except Exception:
                    pass
            if text is not None:
                try:
                    text.set_position((xs[-1], ys[-1] + 2.0))
                except Exception:
                    pass

        # 碰撞点（预测到的第一个碰撞点）
        hit_point = event.get('hit_point')
        if hit_point and isinstance(hit_point, (list, tuple)) and len(hit_point) >= 2:
            try:
                hx, hy = float(hit_point[0]), float(hit_point[1])
                hit_artist = self._avoid_hit_artists.get(vid)
                if hit_artist is None:
                    hit_artist = self.ax.scatter(
                        [hx],
                        [hy],
                        s=120,
                        marker='x',
                        color='yellow',
                        linewidths=3.0,
                        zorder=16,
                    )
                    self._avoid_hit_artists[vid] = hit_artist
                else:
                    hit_artist.set_offsets([(hx, hy)])
            except Exception:
                pass

        # 动态障碍（其他车的膨胀 bbox）：用于可视化“对其他车的安全距离/避让区域”
        try:
            for a in list(self._avoid_dyn_obs_artists):
                try:
                    a.remove()
                except Exception:
                    pass
            self._avoid_dyn_obs_artists = []
        except Exception:
            self._avoid_dyn_obs_artists = []

        try:
            if not bool(self.show_dynamic_obstacle_boxes_var.get()):
                return
        except Exception:
            # 若变量异常，默认显示
            pass

        dyn_obs = event.get('grid_other_vehicle_obstacles') or []
        for i, bbox in enumerate(dyn_obs):
            if not bbox or len(bbox) < 4:
                continue
            try:
                x_min = min(p[0] for p in bbox)
                y_min = min(p[1] for p in bbox)
                width = max(p[0] for p in bbox) - x_min
                height = max(p[1] for p in bbox) - y_min
                rx, ry, rw, rh = self._clamp_bbox(x_min, y_min, width, height)
                rect = Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor='none',
                    edgecolor='#8A2BE2',
                    linewidth=2.4,
                    linestyle='--',
                    alpha=0.85,
                    zorder=11,
                )
                self.ax.add_patch(rect)
                self._avoid_dyn_obs_artists.append(rect)
            except Exception:
                continue

    def _mapper_pkl_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "coordinate_mapper.pkl")

    def _has_coordinate_mapper_pkl(self) -> bool:
        """仅判断 pkl 是否存在。

        注意：不要在这里 import CoordinateMapper，因为它依赖 numpy；GUI 弹窗策略不应依赖运行环境是否装了 numpy。
        """
        try:
            return os.path.exists(self._mapper_pkl_path())
        except Exception:
            return False

    @staticmethod
    def _square_from_bbox(x_min: float, y_min: float, width: float, height: float,
                          xlim=(0.0, 144.0), ylim=(0.0, 108.0)):
        side = float(max(width, height, 1.0))
        cx = float(x_min) + float(width) / 2.0
        cy = float(y_min) + float(height) / 2.0
        sx = cx - side / 2.0
        sy = cy - side / 2.0

        # 尽量保持方框在视野范围内
        try:
            xmin, xmax = float(xlim[0]), float(xlim[1])
            ymin, ymax = float(ylim[0]), float(ylim[1])
            if sx < xmin:
                sx = xmin
            if sy < ymin:
                sy = ymin
            if sx + side > xmax:
                sx = max(xmin, xmax - side)
            if sy + side > ymax:
                sy = max(ymin, ymax - side)
        except Exception:
            pass

        return sx, sy, side

    @staticmethod
    def _clamp_bbox(
        x_min: float,
        y_min: float,
        width: float,
        height: float,
        xlim=(0.0, 144.0),
        ylim=(0.0, 108.0),
        min_size: float = 1.0,
    ):
        """保持原始长方形比例，并尽量把 bbox 限制在视野范围内。"""
        w = float(max(width, min_size))
        h = float(max(height, min_size))
        x = float(x_min)
        y = float(y_min)

        try:
            xmin, xmax = float(xlim[0]), float(xlim[1])
            ymin, ymax = float(ylim[0]), float(ylim[1])

            if x < xmin:
                x = xmin
            if y < ymin:
                y = ymin

            # 尽量不改变 w/h（保持检测比例），只移动位置让其落在边界内
            if x + w > xmax:
                x = max(xmin, xmax - w)
            if y + h > ymax:
                y = max(ymin, ymax - h)
        except Exception:
            pass

        return x, y, w, h

    def _start_realtime_viz_loop(self):
        """定时刷新 ROS 轨迹叠加层。"""

        def tick():
            if self._closing:
                return
            try:
                updated = self.update_real_time_trajectories()
                if updated:
                    self.update_trajectory_visualization()
            except Exception:
                # 可视化循环异常不应影响主线程
                pass

            self._realtime_viz_id = self.root.after(self._realtime_viz_interval_ms, tick)

        tick()

    def _request_canvas_draw(self, min_interval_ms: Optional[int] = None) -> None:
        """合并/节流 Matplotlib 重绘请求。

        Tk + Matplotlib 在多处频繁 draw_idle 时会造成持续重绘与卡顿。
        这里将一段时间内的多次更新合并为一次 draw_idle。
        """
        if getattr(self, '_closing', False):
            return

        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return

        try:
            if not hasattr(self, 'canvas') or self.canvas is None:
                return
        except Exception:
            return

        # 已经排队了就不重复排队（合并）
        if self._draw_after_id is not None:
            return

        interval_ms = int(min_interval_ms) if min_interval_ms is not None else int(getattr(self, '_draw_min_interval_ms', 80))
        interval_s = max(0.0, float(interval_ms) / 1000.0)
        now = time.monotonic()
        due_s = (self._last_draw_ts + interval_s) - now
        delay_ms = int(max(0.0, due_s) * 1000)

        def _do_draw():
            self._draw_after_id = None
            self._last_draw_ts = time.monotonic()
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

        try:
            self._draw_after_id = self.root.after(delay_ms, _do_draw)
        except Exception:
            self._draw_after_id = None

    @staticmethod
    def _uniform_sample_points(points, max_points: int):
        """对点序列做等间隔抽样，最多返回 max_points 个点，覆盖全程。"""
        if not points:
            return []
        if max_points <= 0:
            return []
        n = len(points)
        if n <= max_points:
            return list(points)
        if max_points == 1:
            return [points[0]]

        # n > max_points: 线性等间隔取样
        step = (n - 1) / (max_points - 1)
        indices = [int(i * step) for i in range(max_points - 1)] + [n - 1]
        return [points[i] for i in indices]

    def _start_event_poller(self):
        """在 Tk 主线程中轮询系统事件队列（事件驱动 UI），避免后台线程直接操作 UI。"""
        def poll():
            if self._closing:
                return
            try:
                while True:
                    event = self._event_queue.get_nowait()
                    self._handle_system_event(event)
            except queue.Empty:
                pass
            self._event_poller_id = self.root.after(50, poll)

        poll()

    def _handle_system_event(self, event: dict):
        etype = event.get("type")
        if not etype:
            return

        if etype == "state":
            # 统一走结果区日志，避免频繁弹窗
            state = event.get("state")
            msg = event.get("message", "")
            try:
                now = time.monotonic()
                state_key = f"{state}|{msg}" if msg else f"{state}"
                if state_key == self._last_state_value and (now - self._last_state_log_ts) < 1.0:
                    return
                if (now - self._last_state_log_ts) < 0.35:
                    return
                self._last_state_log_ts = now
                self._last_state_value = state_key
            except Exception:
                pass
            self.update_result_text(f"状态更新: {state} {('- ' + msg) if msg else ''}")
            return

        if etype == "progress":
            progress = event.get("progress")
            msg = event.get("message", "")
            # 高频 progress 事件会导致 ScrolledText 频繁 insert/delete，显著拖慢 UI
            try:
                now = time.monotonic()
                if progress == self._last_progress_value and (now - self._last_progress_log_ts) < 1.0:
                    return
                if (now - self._last_progress_log_ts) < 0.35:
                    return
                self._last_progress_log_ts = now
                self._last_progress_value = progress
            except Exception:
                pass
            self.update_result_text(f"进度: {progress}% {('- ' + msg) if msg else ''}")
            return

        if etype == "error":
            msg = event.get("message", "")
            self.update_result_text(f"错误: {msg}")
            return

        if etype == "running":
            running = bool(event.get("running"))
            self.stop_mission_btn.config(state=(tk.NORMAL if running else tk.DISABLED))

            # 清理避障叠加：放到“任务开始”时做，避免任务结束后画面被清空
            if running:
                try:
                    self._avoidance_state = {}
                    self._clear_avoidance_artists()
                except Exception:
                    pass
            return

        if etype == "avoidance_replanned":
            # 实时避障重规划：叠加显示
            try:
                vid = int(event.get('vehicle_id'))
            except Exception:
                vid = None

            if vid is not None:
                try:
                    self._avoidance_state[vid] = dict(event)
                except Exception:
                    pass
                try:
                    grid_path = event.get('grid_path') or []
                    self.update_result_text(f"避障重规划: vehicle_{vid} 路径点数={len(grid_path)}")
                except Exception:
                    pass

            try:
                self._apply_avoidance_overlay_update(event)
                self._request_canvas_draw()
            except Exception:
                pass
            return

        if etype == "paths_planned":
            grid_path_results = event.get("grid_path_results")
            path_results = event.get("path_results")
            if grid_path_results:
                self.update_path_visualization(path_results, grid_path_results, planned=self.path_planned)
                self.path_planned = True
            return

        if etype == "mission_completed":
            self.stop_mission_btn.config(state=tk.DISABLED)
            self.path_planned = False
            self.update_result_text("任务完成：可以开始下一轮任务")
            return

        if etype == "mission_stopped":
            self.stop_mission_btn.config(state=tk.DISABLED)
            self.path_planned = False
            self.update_result_text("任务已停止：可以重新执行规划")
            return

    def _refresh_visual_after_mission(self, async_snapshot: bool = True) -> None:
        """任务结束后刷新可视化。

        - 背景图：尝试异步拍一张新图（不影响下一轮的坐标映射）
        - 叠加层：重绘规划结果/轨迹（如果仍保留在内存）
        """
        if self._closing:
            return

        def redraw_only():
            try:
                self.update_visualization()
            except Exception:
                pass
            try:
                # 若 trajectories 仍在，重画一遍轨迹叠加层
                self.update_trajectory_visualization()
            except Exception:
                pass

        if not async_snapshot:
            self.root.after(0, redraw_only)
            return

        def worker_capture():
            camera = None
            try:
                camera = Camera(device_index=0)
                try:
                    _ = camera.list_devices()
                except Exception:
                    pass
                if camera.connect():
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"./captures/mission_done_{ts}.jpg"
                    camera.capture_rotated_image(path, angle=self.camera_rotation)
            except Exception:
                pass
            finally:
                try:
                    if camera is not None:
                        camera.disconnect()
                except Exception:
                    pass
                # 无论是否拍照成功，都刷新一次（至少重读 captures 里最新图片）
                try:
                    self.root.after(0, redraw_only)
                except Exception:
                    pass

        threading.Thread(target=worker_capture, daemon=True).start()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        # 左侧面板稍微加宽一点，减少换行/裁切
        control_frame = ttk.Frame(main_frame, width=600)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 控制面板做成可滚动区域：控件多时不挤、文字不被裁切
        control_inner = self._create_scrollable_panel(control_frame)
        
        # 右侧可视化面板
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 控制面板内容
        self.create_control_panel(control_inner)
        
        # 可视化面板内容
        self.create_visualization_panel(viz_frame)

    def _create_scrollable_panel(self, parent: ttk.Frame) -> ttk.Frame:
        """在 parent 内创建一个纵向可滚动容器，返回实际放控件的 inner frame。"""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(event):
            try:
                canvas.itemconfigure(window_id, width=event.width)
            except Exception:
                pass

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # 鼠标滚轮：绑定在面板自身，避免影响全局（尤其是 Matplotlib 的交互）
        def _on_mousewheel(event):
            try:
                if self._closing:
                    return
            except Exception:
                pass
            try:
                delta = int(-1 * (event.delta / 120))
                canvas.yview_scroll(delta, "units")
            except Exception:
                pass

        def _on_mousewheel_linux(event):
            try:
                if self._closing:
                    return
            except Exception:
                pass
            try:
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
            except Exception:
                pass

        def _focus_on_enter(_event=None):
            try:
                canvas.focus_set()
            except Exception:
                pass

        canvas.bind("<Enter>", _focus_on_enter)

        # 同时绑定在 canvas 和 inner，确保鼠标在内容区也能滚动
        for w in (canvas, inner):
            try:
                w.bind("<MouseWheel>", _on_mousewheel)
                w.bind("<Button-4>", _on_mousewheel_linux)
                w.bind("<Button-5>", _on_mousewheel_linux)
            except Exception:
                pass

        return inner
        
    def create_control_panel(self, parent):
        # 文件导入区域
        file_frame = ttk.LabelFrame(parent, text="文件导入", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="导入JSON文件", 
                  command=self.load_json_file).pack(fill=tk.X)
        
        # 车辆配置区域
        config_frame = ttk.LabelFrame(parent, text="车辆配置", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # 用 grid：更省垂直空间；label 给 wraplength 避免裁切
        config_frame.columnconfigure(0, weight=0)
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="车辆ID(逗号):", wraplength=180, justify=tk.LEFT).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.vehicle_ids_entry = tk.Entry(config_frame)
        self.vehicle_ids_entry.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        self.vehicle_ids_entry.insert(0, "0,1,2")

        ttk.Label(config_frame, text="车辆IP(id:ip):", wraplength=180, justify=tk.LEFT).grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.car_ips_entry = tk.Entry(config_frame)
        self.car_ips_entry.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        self.car_ips_entry.insert(0, "0:192.168.1.224,1:192.168.1.227,2:192.168.1.229")

        ttk.Label(config_frame, text="Bias(id:bias):", wraplength=180, justify=tk.LEFT).grid(row=2, column=0, sticky="w", pady=(0, 6))
        self.car_bias_entry = tk.Entry(config_frame)
        self.car_bias_entry.grid(row=2, column=1, sticky="ew", pady=(0, 6))
        self.car_bias_entry.insert(0, "0:0,1:0,2:0")

        ttk.Label(config_frame, text="UDP端口:").grid(row=3, column=0, sticky="w", pady=(0, 6))
        self.car_port_entry = tk.Entry(config_frame)
        self.car_port_entry.grid(row=3, column=1, sticky="ew", pady=(0, 6))
        self.car_port_entry.insert(0, "12345")

        ttk.Button(config_frame, text="应用配置", command=self.apply_vehicle_config).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        
        # 首次检测
        detect_frame = ttk.LabelFrame(parent, text="首次检测", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            detect_frame,
            text="手动映射车辆编号",
            command=self.open_vehicle_mapping_dialog,
        ).pack(fill=tk.X, pady=(0, 6))
        
        self.first_detect_btn = ttk.Button(detect_frame, text="开始首次检测", 
                                        command=self.start_first_detection)
        self.first_detect_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.detection_status = ttk.Label(detect_frame, text="未进行首次检测", 
                                        foreground="orange")
        self.detection_status.pack(pady=(5, 0))
        # 数据显示区域
        data_frame = ttk.LabelFrame(parent, text="数据信息", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.data_text = scrolledtext.ScrolledText(data_frame, height=6)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
        system_frame = ttk.LabelFrame(parent, text="系统控制", padding=10)
        system_frame.pack(fill=tk.X, pady=(0, 10))

        # 注意：ttk.Checkbutton 不支持 wraplength。用“勾选框 + 可换行 Label”组合实现长文本。
        def _wrap_check_row(parent, text, variable, command=None, wrap=470):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=(0, 6), anchor=tk.W)

            cb = ttk.Checkbutton(row, variable=variable, command=command)
            cb.pack(side=tk.LEFT, anchor=tk.N)

            lbl = tk.Label(row, text=text, wraplength=wrap, justify=tk.LEFT, anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # 点击文字也能切换
            def _toggle(_event=None):
                try:
                    variable.set(not bool(variable.get()))
                    if callable(command):
                        command()
                except Exception:
                    pass

            lbl.bind("<Button-1>", _toggle)
            return row

        _wrap_check_row(system_frame, "强制重建坐标映射(pkl)", self.force_rebuild_mapper_var)
        _wrap_check_row(
            system_frame,
            "显示避障重规划叠加(粉色)",
            self.show_avoidance_overlay_var,
            command=lambda: (self._clear_avoidance_artists() if not self.show_avoidance_overlay_var.get() else None),
        )
        _wrap_check_row(system_frame, "任务执行期刷新背景(20Hz，不抓拍)", self.refresh_background_during_run_var)
        _wrap_check_row(system_frame, "显示动态避让框(虚线紫：其他车膨胀障碍)", self.show_dynamic_obstacle_boxes_var)

        self.return_home_btn = ttk.Button(
            system_frame,
            text="回到初始位置",
            command=self.return_to_home,
            state=tk.DISABLED,
        )
        self.return_home_btn.pack(fill=tk.X, pady=(0, 6))

        self.stop_mission_btn = ttk.Button(system_frame, text="停止任务", 
                                        command=self.stop_mission, state=tk.DISABLED)
        self.stop_mission_btn.pack(fill=tk.X)

        # 命令输入区域
        command_frame = ttk.LabelFrame(parent, text="命令输入", padding=10)
        command_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(command_frame, text="输入命令:").pack(anchor=tk.W)
        self.command_entry = tk.Text(command_frame, height=3)
        self.command_entry.pack(fill=tk.X, pady=(5, 0))
        
        # 设置默认命令
        self.command_entry.insert(tk.END, "用车辆0和2包围目的地0;小车0去往目的地0；小车0清扫目的地0")
        
        # 执行按钮
        ttk.Button(command_frame, text="执行路径规划", 
                  command=self.execute_planning).pack(fill=tk.X, pady=(10, 0))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(parent, text="执行结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=6)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def _try_record_home_positions(self) -> None:
        """尝试记录初始位姿（只记录一次）。"""
        if self._closing:
            return

        def worker():
            ok = False
            try:
                ok = bool(self.vehicle_system.record_home_positions(overwrite=False))
            except Exception:
                ok = False

            def apply_ui():
                if ok:
                    try:
                        self.return_home_btn.config(state=tk.NORMAL)
                    except Exception:
                        pass
                    self.update_result_text("已记录初始位置：可一键回归")
            try:
                self.root.after(0, apply_ui)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def return_to_home(self):
        """一键回到初始位置（ROS 坐标）。"""
        if self._closing:
            return

        def worker():
            try:
                ok = bool(self.vehicle_system.start_return_to_home())
                if ok:
                    self.root.after(0, lambda: self.update_result_text("开始回归初始位置"))
                else:
                    self.root.after(0, lambda: self.update_result_text("回归初始位置启动失败（请查看系统状态/ROS位姿）"))
            except Exception as e:
                try:
                    self.root.after(0, lambda: self.update_result_text(f"回归初始位置异常: {str(e)}"))
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

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
            
            # 解析 car_ips（目标：保持 vehicle_id -> ip 映射）
            car_ips_str = self.car_ips_entry.get().strip()
            if car_ips_str:
                car_ips_dict = {}
                for pair in car_ips_str.split(','):
                    if ':' in pair:
                        vehicle_id, ip = pair.split(':', 1)
                        car_ips_dict[int(vehicle_id.strip())] = ip.strip()
                # 直接按 vehicle_id 存储
                self.car_ips = {}
                for i, vehicle_id in enumerate(self.vehicle_ids):
                    if vehicle_id in car_ips_dict:
                        self.car_ips[vehicle_id] = car_ips_dict[vehicle_id]
                    else:
                        self.car_ips[vehicle_id] = f"192.168.1.{200 + i}"
            
            # 解析 car_bias
            car_bias_str = self.car_bias_entry.get().strip()
            if car_bias_str:
                car_bias_dict = {}
                for pair in car_bias_str.split(','):
                    if ':' in pair:
                        vehicle_id, bias = pair.split(':', 1)
                        car_bias_dict[int(vehicle_id.strip())] = float(bias.strip())
                # 直接按 vehicle_id 存储
                self.car_bias = {}
                for i, vehicle_id in enumerate(self.vehicle_ids):
                    if vehicle_id in car_bias_dict:
                        self.car_bias[vehicle_id] = car_bias_dict[vehicle_id]
                    else:
                        self.car_bias[vehicle_id] = 0
            
            # 解析 car_port
            car_port_str = self.car_port_entry.get().strip()
            if car_port_str:
                self.car_port = int(car_port_str)
            
            # 应用配置到系统
            # 直接传递 vehicle_id -> value 的映射
            self.vehicle_system._setup_vehicle_config(
                vehicle_ids=self.vehicle_ids,
                car_ips=self.car_ips,
                car_bias=self.car_bias,
                car_port=self.car_port,
                camera_rotation=self.camera_rotation
            )

            messagebox.showinfo("成功", "车辆配置已应用")
            self.update_result_text("车辆配置已更新")
            
        except Exception as e:
            messagebox.showerror("错误", f"配置应用失败: {str(e)}")

    def start_first_detection(self):
        """开始首次检测"""
        if self._closing:
            return

        # 允许“重新检测”：车辆移动后需要重新生成 detection_results.json
        is_redetect = bool(self.first_detection_done)
        if is_redetect:
            # 不再默认强制重建 mapper.pkl：是否重建由勾选框决定。
            self.yolo_to_real_mapping = {}
            self.trajectories = {}
            self.path_planned = False
            try:
                self.detection_status.config(text="正在重新检测...", foreground="orange")
            except Exception:
                pass
        else:
            try:
                self.detection_status.config(text="正在首次检测...", foreground="orange")
            except Exception:
                pass

        def worker():
            camera = None
            try:
                # 优先尝试使用 system 缓存帧（避免与系统相机连接竞争，导致“未取到画面”）
                frame = None
                try:
                    getter = getattr(self.vehicle_system, 'get_latest_frame', None)
                    if callable(getter):
                        frame = getter(max_age_s=10.0, copy=True)
                except Exception:
                    frame = None

                # 如果系统没有缓存帧，再尝试直接连相机抓一帧
                if frame is None:
                    camera = Camera(device_index=0)
                    # 列出所有设备（如果有方法返回）
                    try:
                        _ = camera.list_devices()
                    except Exception:
                        pass

                    if camera.connect():
                        # 多试几次，兼容相机刚启动/缓冲未就绪
                        for _ in range(3):
                            try:
                                frame = camera.get_frame(timeout=1500, angle=self.camera_rotation)
                            except Exception:
                                frame = None
                            if frame is not None:
                                break

                def on_detect_complete(success: bool):
                    if not success:
                        try:
                            self.detection_status.config(text="检测失败", foreground="red")
                        except Exception:
                            pass
                        return

                    # detect_latest_image 内部已经 load_json_file + update_visualization
                    # 规则：
                    # - 若已有 coordinate_mapper.pkl 且 rotation 匹配，则不强制弹出车辆映射窗口（依赖系统侧 ROS+mapper 自动匹配）
                    # - 若没有 pkl（或用户勾选了“强制重建坐标映射”），则强制弹出窗口让用户提供三车对应关系，以便立刻生成 pkl
                    has_mapper = self._has_coordinate_mapper_pkl()
                    need_mapping = (not has_mapper) or bool(self.force_rebuild_mapper_var.get())
                    if need_mapping:
                        self.open_vehicle_mapping_dialog()
                    else:
                        # 有 pkl 时，自动推断 YOLO->真实车辆ID，用于 UI 标注（不强制用户手动映射）
                        def infer_worker():
                            try:
                                mapping = self.vehicle_system.infer_yolo_to_real_mapping_for_gui(self.grid_vehicles)
                            except Exception:
                                mapping = {}

                            def apply_mapping():
                                if mapping:
                                    self.yolo_to_real_mapping = dict(mapping)
                                    self.update_data_display()
                                    self.update_visualization()
                                    self.update_result_text(f"已自动推断车辆映射: {self.yolo_to_real_mapping}")
                                    # 有映射且有 ROS，可记录初始位姿（只记录一次）
                                    self._try_record_home_positions()
                            try:
                                self.root.after(0, apply_mapping)
                            except Exception:
                                pass

                        threading.Thread(target=infer_worker, daemon=True).start()

                    self.first_detection_done = True
                    self.first_detect_btn.config(text="重新检测")

                    if need_mapping:
                        if is_redetect:
                            self.detection_status.config(text="重新检测完成：请映射车辆编号", foreground="orange")
                            self.update_result_text("重新检测完成：缺少坐标映射器(pkl)，请映射车辆编号以生成 pkl")
                        else:
                            self.detection_status.config(text="首次检测完成：请映射车辆编号", foreground="orange")
                            self.update_result_text("首次检测完成：缺少坐标映射器(pkl)，请映射车辆编号以生成 pkl")
                    else:
                        if is_redetect:
                            self.detection_status.config(text="重新检测完成", foreground="green")
                            self.update_result_text("重新检测完成：已检测到坐标映射器(pkl)，无需强制映射")
                        else:
                            self.detection_status.config(text="首次检测完成", foreground="green")
                            self.update_result_text("首次检测完成：已检测到坐标映射器(pkl)，无需强制映射")

                if frame is not None:
                    # 直接对内存帧做检测（避免 captures 落盘/读回）
                    self.detect_latest_image(frame=frame, verbose=False, on_complete=on_detect_complete)
                else:
                    # 回退：抓拍一张“新文件”再检测（避免误用上次遗留图片）
                    try:
                        os.makedirs("./captures", exist_ok=True)
                    except Exception:
                        pass

                    captured = False
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        p = f"./captures/test_img_{ts}.jpg"
                        if camera is not None:
                            captured = bool(camera.capture_rotated_image(p, angle=self.camera_rotation))
                        else:
                            captured = False
                    except Exception:
                        captured = False

                    if not captured:
                        try:
                            self.root.after(0, lambda: self.update_result_text("检测失败：未获取到画面（可能是相机被系统占用或未启动）"))
                            self.root.after(0, lambda: self.detection_status.config(text="首次检测失败：未取到画面", foreground="red"))
                        except Exception:
                            pass
                        return

                    # 从 captures 最新文件检测（刚刚生成，应当是最新）
                    self.detect_latest_image(frame=None, verbose=False, on_complete=on_detect_complete)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"首次检测失败: {str(e)}"))
                self.root.after(0, lambda: self.detection_status.config(text="首次检测失败", foreground="red"))
            finally:
                try:
                    if camera is not None:
                        camera.disconnect()
                except Exception:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        self._detection_thread = t
        t.start()

    def set_background(self):
        # 优先使用系统检测线程提供的内存帧（避免频繁读 ./captures）
        try:
            if hasattr(self, 'vehicle_system') and self.vehicle_system is not None:
                getter = getattr(self.vehicle_system, 'get_latest_frame', None)
                if callable(getter):
                    frame = getter(max_age_s=10.0, copy=True)
                    if frame is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.background = cv2.resize(rgb, self._bg_display_size, interpolation=cv2.INTER_NEAREST)
                        return
        except Exception:
            pass

        captures_dir = "./captures"
        if not os.path.exists(captures_dir):
            os.makedirs(captures_dir)
            return
        
        # 获取目录中所有图片文件
        image_files = [f for f in os.listdir(captures_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            # 无 captures 时静默返回（优先走内存帧；避免频繁弹窗打扰）
            return
        
        # 按修改时间排序
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(captures_dir, x)))
        
        try:
            selected_image = os.path.join(captures_dir, image_files[-1])
            img = cv2.imread(selected_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.background = cv2.resize(img, self._bg_display_size, interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            messagebox.showerror("错误", f"无法读取图片文件: {str(e)}")
            self.background = None

    def detect_latest_image(self, frame=None, verbose=True, on_complete=None):
        """检测最新的图片。

        Args:
            verbose: 是否输出日志
            on_complete: 可选回调，在主线程完成结果加载后调用，签名 on_complete(success: bool)
        """
        latest_image = None
        if frame is None:
            captures_dir = "./captures"
            if not os.path.exists(captures_dir):
                os.makedirs(captures_dir)
                return

            # 获取目录中所有图片文件
            image_files = [f for f in os.listdir(captures_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                # 无 captures 时静默失败（避免弹窗）；交由调用方决定是否提示
                if callable(on_complete):
                    self.root.after(0, lambda: on_complete(False))
                return

            # 按修改时间排序，获取最新的图片
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(captures_dir, x)))
            latest_image = os.path.join(captures_dir, image_files[-1])
            # 背景预览做缩放，避免大图拖慢 Matplotlib；YOLO 仍用文件路径推理
            try:
                img = cv2.imread(latest_image)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.background = cv2.resize(img, self._bg_display_size, interpolation=cv2.INTER_NEAREST)
                else:
                    self.background = None
            except Exception:
                self.background = None

            # 使用YOLO进行检测（可能耗时）
            try:
                result = predict.detect_objects(latest_image, verbose=verbose)
                result_json = predict.save_detection_results(result, save_dir='./detection_results')
            except Exception as e:
                # 将错误通过主线程显示
                self.root.after(0, lambda: self.update_result_text(f"检测失败: {str(e)}"))
                if callable(on_complete):
                    self.root.after(0, lambda: on_complete(False))
                return
        else:
            # 对内存帧推理（更快，避免磁盘 I/O）
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.background = cv2.resize(rgb, self._bg_display_size, interpolation=cv2.INTER_NEAREST)
            except Exception:
                self.background = None

            try:
                result = predict.detect_objects_from_frame(frame, verbose=verbose)
                result_json = predict.save_detection_results(result, save_dir='./detection_results')
            except Exception as e:
                self.root.after(0, lambda: self.update_result_text(f"检测失败: {str(e)}"))
                if callable(on_complete):
                    self.root.after(0, lambda: on_complete(False))
                return

        if result_json:
            # 载入并更新UI必须在主线程执行
            def apply_results():
                try:
                    self.load_json_file(result_json)
                    self.update_data_display()
                    if verbose:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if latest_image:
                            log_msg = f"{timestamp} 检测到最新图片: {latest_image}\n"
                        else:
                            log_msg = f"{timestamp} 检测到最新画面(内存帧)\n"
                        log_msg += f"车辆: {len(self.grid_vehicles)}, 障碍物: {len(self.grid_obstacles)}, 目的地: {len(self.grid_destinations)}\n"
                        log_msg += f"检测到的车辆ID: {list(range(len(self.grid_vehicles)))}\n"
                        log_msg += f"配置的Vehicle IDs: {self.vehicle_ids}\n"
                        self.update_result_text(log_msg)
                    if callable(on_complete):
                        on_complete(True)
                except Exception as e:
                    self.update_result_text(f"处理检测结果失败: {str(e)}")
                    if callable(on_complete):
                        on_complete(False)

            self.root.after(0, apply_results)
    
    def open_vehicle_mapping_dialog(self):
        """打开车辆编号映射对话框"""
        if len(self.grid_vehicles) == 0:
            messagebox.showwarning("警告", "未检测到车辆", parent=self.root)
            return
        
        # 创建映射窗口
        mapping_window = tk.Toplevel(self.root)
        mapping_window.title("车辆编号映射")
        mapping_window.geometry("700x600")
        self._child_windows.add(mapping_window)

        # 作为对话框：置顶/聚焦/抓取，避免后续 messagebox 跑到后面导致无法点击
        try:
            mapping_window.transient(self.root)
        except Exception:
            pass
        try:
            mapping_window.lift()
            mapping_window.focus_force()
        except Exception:
            pass
        try:
            mapping_window.grab_set()
        except Exception:
            pass
        
        # 创建说明标签
        info_frame = ttk.Frame(mapping_window)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            info_frame,
            text="给每个 YOLO 框填真实车辆ID（ROS vehicle_x）。保存后会生成/更新 coordinate_mapper.pkl。",
            font=("微软雅黑", 10)
        ).pack()
        ttk.Label(info_frame, text="真实车辆编号应为: " + str(self.vehicle_ids), 
                 font=("微软雅黑", 9, "italic")).pack(pady=(5, 0))
        
        # 创建主框架
        main_frame = ttk.Frame(mapping_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：图形显示
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        fig, ax = plt.subplots(figsize=(5, 4.5))
        canvas = FigureCanvasTkAgg(fig, left_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 关闭处理：确保 matplotlib figure 被关闭，避免残留事件源
        canvas_click_cid = None
        def close_mapping_window():
            try:
                if canvas_click_cid is not None:
                    canvas.mpl_disconnect(canvas_click_cid)
            except Exception:
                pass
            try:
                mapping_window.grab_release()
            except Exception:
                pass
            try:
                plt.close(fig)
            except Exception:
                pass
            try:
                if mapping_window in self._child_windows:
                    self._child_windows.remove(mapping_window)
            except Exception:
                pass
            try:
                mapping_window.destroy()
            except Exception:
                pass

        mapping_window.protocol("WM_DELETE_WINDOW", close_mapping_window)
        
        # 绘制检测到的车辆（保持原始长方形比例）
        rectangles = []
        for i, vehicle in enumerate(self.grid_vehicles):
            if len(vehicle) >= 4:
                x_min = min(point[0] for point in vehicle)
                y_min = min(point[1] for point in vehicle)
                width = max(point[0] for point in vehicle) - x_min
                height = max(point[1] for point in vehicle) - y_min

                rx, ry, rw, rh = self._clamp_bbox(x_min, y_min, width, height)

                rect = Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor='none',
                    edgecolor='deepskyblue',
                    linewidth=3,
                )
                ax.add_patch(rect)
                
                # 添加YOLO索引标签
                center_x = rx + rw / 2
                center_y = ry + rh / 2
                ax.text(center_x, center_y, f'Y{i}',
                       ha='center', va='center', fontweight='bold', fontsize=12,
                       color='black',
                       bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2'))
                
                # 保存矩形信息用于点击
                rectangles.append({
                    'index': i,
                    'x_min': rx, 'y_min': ry,
                    'width': rw, 'height': rh,
                    'rect': rect
                })
        
        # 设置背景
        if hasattr(self, 'background') and self.background is not None:
            ax.imshow(self.background, extent=[0, 144, 0, 108], alpha=0.5, zorder=0)
        
        ax.set_xlim(0, 144)
        ax.set_ylim(0, 108)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        canvas.draw()
        
        # 右侧：映射编辑区域
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        ttk.Label(right_frame, text="映射关系", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W)
        
        # 映射输入框列表
        mapping_entries = {}
        mapping_frame = ttk.Frame(right_frame)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        for i in range(len(self.grid_vehicles)):
            frame = ttk.Frame(mapping_frame)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=f"YOLO{i} →", width=8).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=8)
            entry.pack(side=tk.LEFT, padx=(5, 0))
            
            # 如果已经有映射，则显示
            if i in self.yolo_to_real_mapping:
                entry.insert(0, str(self.yolo_to_real_mapping[i]))
            
            mapping_entries[i] = entry
        
        # 点击处理
        def on_canvas_click(event):
            if event.xdata is None or event.ydata is None:
                return
            
            for rect_info in rectangles:
                x_min = rect_info['x_min']
                y_min = rect_info['y_min']
                width = rect_info['width']
                height = rect_info['height']
                
                if (x_min <= event.xdata <= x_min + width and 
                    y_min <= event.ydata <= y_min + height):
                    # 弹出输入框
                    idx = rect_info['index']
                    dialog = tk.simpledialog.askstring(
                        "输入车辆编号",
                        f"请输入YOLO检测{idx}对应的真实车辆编号\n(可选值: {self.vehicle_ids})",
                        parent=mapping_window,
                    )
                    if dialog:
                        try:
                            real_id = int(dialog)
                            mapping_entries[idx].delete(0, tk.END)
                            mapping_entries[idx].insert(0, str(real_id))
                        except:
                            messagebox.showerror("错误", "请输入有效的整数", parent=mapping_window)
                    break
        
        # 导入 simpledialog
        import tkinter.simpledialog
        canvas_click_cid = canvas.mpl_connect('button_press_event', on_canvas_click)
        
        # 保存和关闭按钮
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_mapping():
            """保存映射关系"""
            try:
                self.yolo_to_real_mapping = {}
                for i, entry in mapping_entries.items():
                    value = entry.get().strip()
                    if value:
                        real_id = int(value)
                        if real_id in self.vehicle_ids:
                            self.yolo_to_real_mapping[i] = real_id
                        else:
                            raise ValueError(f"车辆ID {real_id} 不在配置的车辆列表中")
                
                if len(self.yolo_to_real_mapping) != len(self.grid_vehicles):
                    raise ValueError("请为所有检测到的车辆设置映射关系")

                # 先把映射同步给系统（用于建立/更新 coordinate_mapper.pkl）
                try:
                    self.vehicle_system.set_manual_vehicle_mapping(self.yolo_to_real_mapping)
                except Exception:
                    pass

                # 异步生成/更新坐标映射器 pkl：不等到“执行规划”才生成
                try:
                    self.detection_status.config(text="正在生成坐标映射(pkl)...", foreground="orange")
                except Exception:
                    pass

                def build_mapper_worker():
                    ok = False
                    try:
                        # 手动映射的语义就是“建立/覆盖坐标映射器(pkl)”，不应依赖额外勾选框
                        try:
                            self.vehicle_system.set_force_rebuild_mapper(True)
                        except Exception:
                            pass
                        ok = bool(self.vehicle_system.prepare_coordinate_mapping_from_manual_mapping())
                    except Exception:
                        ok = False

                    def finish_ui():
                        if ok:
                            self.update_result_text("坐标映射器(pkl)已就绪")
                            try:
                                self.detection_status.config(text="映射已保存，pkl已生成", foreground="green")
                            except Exception:
                                pass
                            # 已经生成过 pkl，默认把“强制重建”勾选复位，避免后续误触发
                            try:
                                self.force_rebuild_mapper_var.set(False)
                            except Exception:
                                pass
                            # 手动映射 + pkl 生成成功后，记录一次初始位姿
                            self._try_record_home_positions()
                        else:
                            self.update_result_text("坐标映射器(pkl)生成失败：可稍后在启动任务时再生成，或检查 ROS/车辆位姿")
                            try:
                                self.detection_status.config(text="映射已保存，但pkl生成失败", foreground="orange")
                            except Exception:
                                pass

                        self.update_data_display()  # 更新显示，使用真实车辆ID
                        self.update_visualization()
                        try:
                            mapping_window.lift()
                            mapping_window.focus_force()
                        except Exception:
                            pass
                        messagebox.showinfo("成功", "映射已保存", parent=mapping_window)
                        close_mapping_window()

                    try:
                        self.root.after(0, finish_ui)
                    except Exception:
                        pass

                threading.Thread(target=build_mapper_worker, daemon=True).start()
                
            except Exception as e:
                messagebox.showerror("错误", f"保存映射失败: {str(e)}", parent=mapping_window)
        
        ttk.Button(button_frame, text="保存映射", command=save_mapping).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="关闭", command=close_mapping_window).pack(fill=tk.X)

    def start_mission(self):
        """启动任务"""
        # 检查是否正在关闭
        if self._closing:
            return
        
        if not self.grid_vehicles or not self.grid_destinations:
            messagebox.showwarning("警告", "请先导入数据")
            return
        
        command = self.command_entry.get(1.0, tk.END).strip()
        if not command:
            messagebox.showwarning("警告", "请输入命令")
            return

        # 将手动映射同步给系统侧，保证规划结果回到正确的 ROS vehicle_x
        try:
            if self.yolo_to_real_mapping:
                self.vehicle_system.set_manual_vehicle_mapping(self.yolo_to_real_mapping)
        except Exception:
            pass

        # 是否强制重建坐标映射器（仅影响下一次任务启动）
        try:
            self.vehicle_system.set_force_rebuild_mapper(bool(self.force_rebuild_mapper_var.get()))
        except Exception:
            pass

        # 异步启动，避免阻塞 Tk 主线程（否则停止按钮点不到）
        self.stop_mission_btn.config(state=tk.NORMAL)
        def worker():
            try:
                ok = self.vehicle_system.start_mission(command)
                self.root.after(0, lambda: self.update_result_text(
                    f"任务启动{'成功' if ok else '失败'}: {command}" if ok else "任务启动失败"
                ))
                if not ok:
                    self.root.after(0, lambda: self.stop_mission_btn.config(state=tk.DISABLED))
            except Exception as e:
                if not self._closing:
                    self.root.after(0, lambda: messagebox.showerror("错误", f"启动任务失败: {str(e)}"))
                self.root.after(0, lambda: self.stop_mission_btn.config(state=tk.DISABLED))

        threading.Thread(target=worker, daemon=True).start()

    def stop_mission(self):
        """停止任务"""
        try:
            # 检查是否正在关闭
            if self._closing:
                return

            # 异步停止，避免 join 阻塞 UI
            def worker():
                try:
                    self.vehicle_system.stop_mission()
                finally:
                    self.root.after(0, lambda: self.stop_mission_btn.config(state=tk.DISABLED))
                    self.path_planned = False

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            if not self._closing:
                messagebox.showerror("错误", f"停止任务失败: {str(e)}")
    
    def update_trajectory_visualization(self):
        """更新轨迹可视化"""
        if not self.trajectories:
            return

        # 复用/更新轨迹 artist，避免频繁 remove/add 造成卡顿
        # 移除已不存在的车辆 artist
        try:
            existing_vids = set(self._traj_artists.keys())
            current_vids = set(self.trajectories.keys())
            for vid in list(existing_vids - current_vids):
                artists = self._traj_artists.pop(vid, None)
                if not artists:
                    continue
                for k in ("line", "point", "text", "start"):
                    a = artists.get(k)
                    if a is None:
                        continue
                    try:
                        a.remove()
                    except Exception:
                        pass
        except Exception:
            pass

        # 创建反向映射：真实ID -> YOLO索引
        real_to_yolo = {v: k for k, v in self.yolo_to_real_mapping.items()} if self.yolo_to_real_mapping else {}

        for i, (vid, traj) in enumerate(self.trajectories.items()):
            if len(traj) < 2:
                continue
            # 去重（避免重复点导致“抖动/回连”）
            clean_traj = []
            prev_point = None
            for p in traj:
                if p != prev_point:
                    clean_traj.append(p)
                    prev_point = p
            traj = clean_traj
            if len(traj) < 2:
                continue

            # 循环使用颜色方案
            color_idx = vid % len(self.color_palette)
            line_color, point_color = self.color_palette[color_idx]
            
            # 绘制轨迹线
            x_vals = [p[0] for p in traj]
            y_vals = [p[1] for p in traj]
            
            # 创建标签：显示真实ID，括号注明YOLO编号
            yolo_idx = real_to_yolo.get(vid)
            if yolo_idx is not None and self.yolo_to_real_mapping:
                trajectory_label = f'车辆{vid}轨迹(Y{yolo_idx})'
            else:
                trajectory_label = f'车辆{vid}轨迹'
            
            last_point = traj[-1]
            start_point = traj[0]

            artists = self._traj_artists.get(vid)
            if not artists:
                # 首次创建
                line = self.ax.plot(
                    x_vals,
                    y_vals,
                    color=line_color,
                    linestyle='-',
                    linewidth=4.5,
                    alpha=0.85,
                    marker='',
                    zorder=15,
                )[0]
                point = self.ax.scatter(
                    [last_point[0]],
                    [last_point[1]],
                    color=point_color,
                    s=180,
                    edgecolors='white',
                    linewidths=2.0,
                    zorder=20,
                    alpha=0.95,
                )
                start_scatter = self.ax.scatter(
                    [start_point[0]],
                    [start_point[1]],
                    color=line_color,
                    s=130,
                    marker='*',
                    edgecolors='gold',
                    linewidths=1.5,
                    zorder=14,
                )

                if yolo_idx is not None and self.yolo_to_real_mapping:
                    label_text = f'🚗 {vid}\n(Y{yolo_idx})'
                else:
                    label_text = f'🚗 {vid}'

                text = self.ax.text(
                    last_point[0],
                    last_point[1] + 2.4,
                    label_text,
                    color='white',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    zorder=21,
                    bbox=dict(
                        facecolor=line_color,
                        alpha=0.75,
                        boxstyle='round,pad=0.35',
                        edgecolor='white',
                    ),
                )

                self._traj_artists[vid] = {
                    "line": line,
                    "point": point,
                    "text": text,
                    "start": start_scatter,
                }
            else:
                # 更新已有 artist
                line = artists.get("line")
                point = artists.get("point")
                text = artists.get("text")
                start_scatter = artists.get("start")

                if line is not None:
                    line.set_data(x_vals, y_vals)

                if point is not None:
                    try:
                        point.set_offsets([[last_point[0], last_point[1]]])
                    except Exception:
                        pass

                if start_scatter is not None:
                    try:
                        start_scatter.set_offsets([[start_point[0], start_point[1]]])
                    except Exception:
                        pass

                if text is not None:
                    if yolo_idx is not None and self.yolo_to_real_mapping:
                        label_text = f'🚗 {vid}\n(Y{yolo_idx})'
                    else:
                        label_text = f'🚗 {vid}'
                    try:
                        text.set_text(label_text)
                        text.set_position((last_point[0], last_point[1] + 2.4))
                    except Exception:
                        pass
        
        # 请求重绘（合并/节流）
        self._request_canvas_draw()
                
    def monitor_path_results(self):
        """持续监听系统的路径规划结果"""
        # 旧的轮询逻辑已被事件驱动替代，保留空实现以兼容历史调用。
        return
        if not hasattr(self, '_monitor_running'):
            self._monitor_running = True  # 监控运行标志
        
        def check_results():
            if not (self._monitor_running and self.vehicle_system.running):
                return
                
            try:
                # 批量获取需要的数据，减少属性访问次数
                system_data = {
                    'running': self.vehicle_system.running,
                    'path_results': getattr(self.vehicle_system, 'path_results', None),
                    'grid_path_results': getattr(self.vehicle_system, 'grid_path_results', None),
                    'mission_completed': getattr(self.vehicle_system, 'mission_completed', False)
                }
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self._update_ui_with_results(system_data))
                
            except Exception as e:
                self.root.after(0, lambda: self.update_result_text(f"监控错误: {str(e)}"))
            
            # 使用after循环而非递归
            if self._monitor_running:
                self._monitor_id = self.root.after(1000, check_results)  # 统一1000ms间隔
        
        # 启动监控
        check_results()

    def _update_ui_with_results(self, system_data):
        """在主线程中安全更新UI"""
        try:
            # 1. 更新基础可视化
            self.load_json_file('./detection_results/detection_results.json')
            self.update_visualization()
            
            # 2. 如果有新路径结果，更新路径可视化
            if system_data['grid_path_results']:
                self.update_path_visualization(
                    system_data['path_results'],
                    system_data['grid_path_results'],
                    planned=self.path_planned
                )
                self.path_planned = True

            # 3. 更新实时轨迹
            # self.update_real_time_trajectories()
            print("可视化已更新")
        except Exception as e:
            self.update_result_text(f"UI更新失败: {str(e)}")

    def stop_monitoring(self):
        """停止监控"""
        if hasattr(self, '_monitor_running'):
            self._monitor_running = False
        if hasattr(self, '_monitor_id'):
            self.root.after_cancel(self._monitor_id)

    def update_path_visualization(self, path_results, grid_path_results, planned=False):
            """更新路径可视化"""
            try:
                # self.detect_latest_image(verbose=False)

                # 绘制路径
                if grid_path_results:
                    self.draw_path_on_ax(grid_path_results)
                    if not planned:
                        result_text = "路径规划完成：\n" + "\n".join(str(res) for res in grid_path_results)
                        self.result_text.insert(tk.END, result_text + "\n")
                    self._request_canvas_draw()
                
            except Exception as e:
                self.update_result_text(f"可视化更新失败: {str(e)}") 

    def update_real_time_trajectories(self):
        """更新实时轨迹显示"""
        if self._closing:
            return False

        if not hasattr(self, 'vehicle_system') or not self.vehicle_system.running:
            return False
        
        # 获取所有车辆的实时轨迹（来自控制器轨迹历史，通常是“从起点到当前”的全量列表）
        trajectories_updated = False
        for vid in self.vehicle_ids:
            full_traj = self.vehicle_system.get_actual_trajectory(vid)
            if not full_traj:
                continue

            # 过滤明显无效的位姿点（常见：VRPN/ROS 未就绪时 (0,0) 或 NaN/Inf）
            raw = []
            for p in full_traj:
                try:
                    x, y = float(p[0]), float(p[1])
                except Exception:
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                if abs(x) < 1e-6 and abs(y) < 1e-6:
                    continue
                raw.append((x, y))

            if len(raw) < 2:
                continue

            # 对全程等间隔抽样：最多 250 个点，保证能涵盖整条路线
            MAX_SAMPLE_POINTS = 250
            raw = self._uniform_sample_points(raw, MAX_SAMPLE_POINTS)

            # 转换坐标到图像->网格坐标系
            if hasattr(self.vehicle_system, "mapper") and self.vehicle_system.mapper and self.vehicle_system.mapper.is_initialized:
                try:
                    img_points = self.vehicle_system.mapper.batch_map_to_image_coords(raw)
                    grid_points = batch_convert_to_grid_coordinates(img_points)
                    traj = grid_points
                except Exception:
                    traj = []
            else:
                traj = []

            if traj:
                # 过滤明显跑到图外的点，避免出现“从很远的地方连一条线过来”
                # （图的范围是 0..144, 0..108，这里加一点 margin）
                xmin, xmax = -5, 149
                ymin, ymax = -5, 113
                filtered = []
                for p in traj:
                    try:
                        gx, gy = float(p[0]), float(p[1])
                    except Exception:
                        continue
                    if not (math.isfinite(gx) and math.isfinite(gy)):
                        continue
                    if gx < xmin or gx > xmax or gy < ymin or gy > ymax:
                        continue
                    filtered.append((gx, gy))

                # 跳变点剔除：如果相邻点距离过大，丢弃该异常点
                cleaned = []
                JUMP_TH = 25.0  # 网格坐标下的跳变阈值（可按需要调整）
                for p in filtered:
                    if not cleaned:
                        cleaned.append(p)
                        continue
                    dx = p[0] - cleaned[-1][0]
                    dy = p[1] - cleaned[-1][1]
                    if (dx * dx + dy * dy) ** 0.5 > JUMP_TH:
                        continue
                    cleaned.append(p)

                if len(cleaned) >= 2:
                    self.trajectories[vid] = cleaned
                    trajectories_updated = True

        return trajectories_updated

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
            # 获取真实车辆ID
            real_id = self.yolo_to_real_mapping.get(i, i) if self.yolo_to_real_mapping else i
            if self.yolo_to_real_mapping:
                info += f"  车辆{real_id} (YOLO{i}): {vehicle}\n"
            else:
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
        self.set_background()
        if hasattr(self, 'background') and self.background is not None:
            self._bg_artist = self.ax.imshow(self.background, extent=[0, 144, 0, 108], alpha=0.7, zorder=0)
        else:
            self._bg_artist = None
        
        if not self.grid_vehicles and not self.grid_obstacles and not self.grid_destinations:
            self.ax.text(0.5, 0.5, "请导入JSON文件", 
                        ha='center', va='center', transform=self.ax.transAxes)
            self._request_canvas_draw(min_interval_ms=0)
            return
        
        # 绘制障碍物（红色框：保持原始长方形比例）
        for i, obstacle in enumerate(self.grid_obstacles):
            if len(obstacle) >= 4:
                x_min = min(point[0] for point in obstacle)
                y_min = min(point[1] for point in obstacle)
                width = max(point[0] for point in obstacle) - x_min
                height = max(point[1] for point in obstacle) - y_min

                rx, ry, rw, rh = self._clamp_bbox(x_min, y_min, width, height)

                rect = Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor='none',
                    edgecolor='red',
                    linewidth=3,
                    zorder=5,
                )
                self.ax.add_patch(rect)
                
                # 添加障碍物标签
                center_x = rx + rw / 2
                center_y = ry + rh / 2
                self.ax.text(center_x, center_y, f'O{i}',
                           ha='center', va='center', fontweight='bold', zorder=6,
                           fontsize=12,
                           bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # 绘制目的地（绿色框：保持原始长方形比例）
        for i, dest in enumerate(self.grid_destinations):
            if len(dest) >= 4:
                x_min = min(point[0] for point in dest)
                y_min = min(point[1] for point in dest)
                width = max(point[0] for point in dest) - x_min
                height = max(point[1] for point in dest) - y_min

                rx, ry, rw, rh = self._clamp_bbox(x_min, y_min, width, height)

                rect = Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor='none',
                    edgecolor='green',
                    linewidth=3,
                    zorder=5,
                )
                self.ax.add_patch(rect)
                
                # 添加目的地标签
                center_x = rx + rw / 2
                center_y = ry + rh / 2
                self.ax.text(center_x, center_y, f'D{i}',
                           ha='center', va='center', fontweight='bold', zorder=6,
                           fontsize=12,
                           bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2'))
                
        # 绘制车辆（蓝色框：保持原始长方形比例 + 更清晰的标签）
        for i, vehicle in enumerate(self.grid_vehicles):
            if len(vehicle) >= 4:
                x_min = min(point[0] for point in vehicle)
                y_min = min(point[1] for point in vehicle)
                width = max(point[0] for point in vehicle) - x_min
                height = max(point[1] for point in vehicle) - y_min

                rx, ry, rw, rh = self._clamp_bbox(x_min, y_min, width, height)

                rect = Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor='none',
                    edgecolor='deepskyblue',
                    linewidth=3,
                    zorder=5,
                )
                self.ax.add_patch(rect)
                
                # 添加车辆标签 - 使用真实车辆ID
                real_id = self.yolo_to_real_mapping.get(i, i) if self.yolo_to_real_mapping else i
                center_x = rx + rw / 2
                if self.yolo_to_real_mapping:
                    label = f'V{real_id} (Y{i})'
                else:
                    label = f'Y{i}'

                # 标签放在方框上沿附近，避免遮挡中心
                self.ax.text(
                    center_x,
                    ry + rh + 1.6,
                    label,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    zorder=6,
                    fontsize=12,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.25'),
                )
        
        # 设置图形属性
        self.ax.set_xlim(0, 144)
        self.ax.set_ylim(0, 108)
        self.ax.set_xlabel('X坐标', fontsize=12)
        self.ax.set_ylabel('Y坐标', fontsize=12)
        self.ax.set_title('车辆路径规划可视化', fontsize=14, fontweight='bold')
        try:
            self.ax.tick_params(axis='both', which='major', labelsize=11)
        except Exception:
            pass
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='deepskyblue', linewidth=3, label='车辆'),
            plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='red', linewidth=3, label='障碍物'),
            plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='green', linewidth=3, label='目的地')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', prop={'size': 12})

        # 避障叠加（若有缓存状态，重建一遍 artist）
        try:
            self._clear_avoidance_artists()
            if bool(self.show_avoidance_overlay_var.get()) and self._avoidance_state:
                for _vid, _evt in list(self._avoidance_state.items()):
                    try:
                        self._apply_avoidance_overlay_update(_evt)
                    except Exception:
                        continue
        except Exception:
            pass
        
        self._request_canvas_draw(min_interval_ms=0)

    def draw_path_on_ax(self, path_list):
        """将路径绘制到已有的 self.ax 上
        
        Args:
            path_list: 路径列表，每个路径是点坐标列表或包含路径数据的字典
        """
        if not path_list:
            return
        
        # 创建反向映射：真实ID -> YOLO索引
        real_to_yolo = {v: k for k, v in self.yolo_to_real_mapping.items()} if self.yolo_to_real_mapping else {}
        
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
            self.ax.plot(
                path_x,
                path_y,
                color='orange',
                linewidth=3.5,
                linestyle='--',
                label='规划路径',
                marker='o',
                zorder=8,
                markersize=6,
                alpha=0.9,
            )

            # 标注路径点序号
            for i, (x, y) in enumerate(path):
                self.ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                                fontsize=10, color='red', weight='bold', zorder=9,
                                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))

            # 起点和终点标记
            start_point = path[0]
            end_point = path[-1]
            self.ax.plot(start_point[0], start_point[1], 'go', markersize=14, label='起点', zorder=10)
            self.ax.plot(end_point[0], end_point[1], 'ro', markersize=14, label='终点', zorder=10)

            # 添加车辆编号标签 - 显示真实ID，括号注明YOLO编号
            if vehicle_index is not None:
                yolo_idx = real_to_yolo.get(vehicle_index)
                if yolo_idx is not None and self.yolo_to_real_mapping:
                    vehicle_label = f'车辆{vehicle_index}(Y{yolo_idx})'
                else:
                    vehicle_label = f'车辆{vehicle_index}'
                
                self.ax.text(
                    start_point[0],
                    start_point[1] + 1.6,
                    vehicle_label,
                    fontsize=11,
                    color='black',
                    fontweight='bold',
                    zorder=11,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='yellow', alpha=0.7),
                )

    def execute_planning(self):
        """执行路径规划"""
        if self._closing:
            return
        
        if not self.grid_vehicles or not self.grid_destinations:
            default_json = './detection_results/detection_results.json'
            if os.path.exists(default_json):
                try:
                    self.load_json_file(default_json)
                except Exception:
                    pass
        if not self.grid_vehicles or not self.grid_destinations:
            messagebox.showwarning("警告", "请先检测/导入包含车辆和目的地数据的JSON文件")
            return
        
        command = self.command_entry.get(1.0, tk.END).strip()
        if not command:
            messagebox.showwarning("警告", "请输入命令")
            return
        
        self.start_mission()

    def setup_system_callbacks(self):
        """设置系统回调函数"""
        def event_callback(event: dict):
            # 系统线程/ROS 回调线程中推送事件；GUI 主线程统一消费
            if not self._closing:
                self._event_queue.put(event)

        # 系统侧已经会通过 event_callback 推送 state/progress/error，
        # 这里避免再用 root.after 直接写 UI，防止窗口销毁后触发 TclError。
        self.vehicle_system.set_callbacks(None, None, None, event_callback)

    def update_result_text(self, message):
        """线程安全的结果文本更新"""
        if self._closing:
            return
        try:
            if not self.root.winfo_exists():
                return
            if not hasattr(self, 'result_text') or not self.result_text.winfo_exists():
                return
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.result_text.insert(tk.END, f"[{timestamp}] {message}\n")

            # 防止日志无限增长导致 UI 越来越卡：保留最近 N 行
            try:
                max_lines = 300
                end_index = self.result_text.index('end-1c')
                total_lines = int(str(end_index).split('.')[0])
                if total_lines > max_lines:
                    # 删除多余行（从第 1 行到超出的那一行）
                    self.result_text.delete('1.0', f"{total_lines - max_lines}.0")
            except Exception:
                pass

            self.result_text.see(tk.END)
        except tk.TclError:
            # 窗口正在销毁或控件已销毁
            return

    def on_closing(self):
        """处理窗口关闭事件，清理所有资源"""
        try:
            # 立即设置关闭标志，防止进一步操作
            self._closing = True

            # 0. 解除系统回调，避免在销毁后继续调 Tk
            try:
                if self.vehicle_system:
                    self.vehicle_system.set_callbacks(None, None, None, None)
            except Exception:
                pass

            # 0.1 先退出 Tk 主循环，避免 UI 线程被后续清理卡住
            try:
                self.root.quit()
            except Exception:
                pass

            # 0.2 关闭所有子窗口（例如“车辆编号映射”窗口）
            try:
                for w in list(self._child_windows):
                    try:
                        w.destroy()
                    except Exception:
                        pass
                self._child_windows.clear()
            except Exception:
                pass

            # 1. 停止事件轮询
            if self._event_poller_id is not None:
                try:
                    self.root.after_cancel(self._event_poller_id)
                except Exception:
                    pass

            # 1.1 停止实时可视化轮询
            if getattr(self, '_realtime_viz_id', None) is not None:
                try:
                    self.root.after_cancel(self._realtime_viz_id)
                except Exception:
                    pass

            # 1.2 取消待执行的重绘回调（如果有）
            if getattr(self, '_draw_after_id', None) is not None:
                try:
                    self.root.after_cancel(self._draw_after_id)
                except Exception:
                    pass
                try:
                    self._draw_after_id = None
                except Exception:
                    pass

            # 2. 停止任务并清理系统资源（确保线程/相机/控制连接收尾）
            try:
                if self.vehicle_system and not self._system_cleaned:
                    # cleanup 可能包含阻塞调用；放到守护线程，避免关闭卡死
                    def cleanup_worker():
                        try:
                            self.vehicle_system.cleanup()
                        except Exception:
                            pass
                    threading.Thread(target=cleanup_worker, daemon=True).start()
                    self._system_cleaned = True
            except Exception:
                pass

            # 3. 等待首次检测线程退出（避免后台线程残留）
            try:
                if self._detection_thread and self._detection_thread.is_alive():
                    self._detection_thread.join(timeout=2)
            except Exception:
                pass
            
            print("程序正在关闭...")
            
        except Exception as e:
            print(f"关闭时出错: {e}")
        finally:
            # 销毁窗口
            try:
                self.root.destroy()
            except:
                pass

            # 兜底：如果还有不可控的非守护线程导致进程不退出，延迟强制退出
            try:
                killer = threading.Timer(5.0, lambda: os._exit(0))
                killer.daemon = True
                killer.start()
            except Exception:
                pass

            # GUI 本地资源清理
            self._cleanup_resources()

    def _cleanup_resources(self):
        """异步清理资源，避免Python shutdown问题"""
        try:
            # 1. 关闭所有 matplotlib 图形（包括子窗口的 figure）
            try:
                plt.close('all')
            except Exception:
                pass
            
            # 2. 清空数据
            try:
                self.grid_vehicles.clear()
                self.grid_obstacles.clear()
                self.grid_destinations.clear()
                self.trajectories.clear()
                self.yolo_to_real_mapping.clear()
            except:
                pass
            
            # 3. 最后清理车辆系统（避免在Python shutdown期间调用）
            # 注意：系统 cleanup 已在 on_closing 中完成，避免重复 cleanup 触发二次 stop/事件。
                
        except:
            pass

    def __del__(self):
        """析构函数"""
        # Tk 对象析构阶段不保证线程/事件安全，避免在此处再触发 cleanup。
        return

def main():
    root = tk.Tk()
    app = VehiclePlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# export ROS_MASTER_URI=http://192.168.1.100:11311
# roslaunch vrpn_client_ros sample.launch server:=192.168.0.76
# /opt/MVS/bin/MVS.sh