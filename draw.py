import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_path(input_data):
    """可视化路径规划数据"""
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制车辆
    vehicles = input_data.get("all_vehicle", [])
    for i, vehicle_coords in enumerate(vehicles):
        vehicle = Polygon(vehicle_coords, fill=True, color='blue', alpha=0.6)
        ax.add_patch(vehicle)
        # 添加车辆标签
        centroid = np.mean(vehicle_coords, axis=0)
        ax.text(centroid[0], centroid[1], f'车{i+1}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    # 绘制障碍物
    obstacles = input_data.get("obstacle", [])
    for obstacle_coords in obstacles:
        obstacle = Polygon(obstacle_coords, fill=True, color='red', alpha=0.7)
        ax.add_patch(obstacle)
    
    # 绘制目的地
    destination_coords = input_data.get("destination", [])
    if destination_coords:
        destination = Polygon(destination_coords, fill=True, color='green', alpha=0.6)
        ax.add_patch(destination)
        # 添加目的地标签
        centroid = np.mean(destination_coords, axis=0)
        ax.text(centroid[0], centroid[1], '目标', 
                ha='center', va='center', color='white', fontweight='bold')
    
    # 绘制路径
    paths = input_data.get("path", [])
    path_colors = ['orange', 'purple', 'brown', 'cyan', 'magenta']
    
    for i, path in enumerate(paths):
        if not path:  # 跳过空路径
            continue
            
        # 提取x和y坐标
        x = [point[0] for point in path]
        y = [point[1] for point in path]
        
        # 选择颜色
        color = path_colors[i % len(path_colors)]
        
        # 绘制路径线
        ax.plot(x, y, 'o-', color=color, linewidth=2, markersize=6, 
                label=f'路径{i+1}')
        
        # 为路径起点添加标签
        if path:
            ax.text(x[0], y[0], f'起{i+1}', 
                    ha='right', va='bottom', color=color, fontweight='bold')
            # 为路径终点添加标签
            ax.text(x[-1], y[-1], f'终{i+1}', 
                    ha='left', va='top', color=color, fontweight='bold')
    
    # 设置坐标轴范围，确保所有元素都能显示
    all_coords = []
    for key in ["vehicle", "obstacle", "destination", "path"]:
        if key in input_data:
            if key == "path":
                for path in input_data[key]:
                    # 过滤掉非坐标对元素
                    valid_points = [p for p in path if isinstance(p, (list, tuple)) and len(p) >= 2]
                    all_coords.extend(valid_points)
            else:
                for item in input_data[key]:
                    # 过滤掉非坐标对元素
                    valid_points = [p for p in item if isinstance(p, (list, tuple)) and len(p) >= 2]
                    all_coords.extend(valid_points)
    
    if all_coords:
        min_x = min(coord[0] for coord in all_coords) - 1
        max_x = max(coord[0] for coord in all_coords) + 1
        min_y = min(coord[1] for coord in all_coords) - 1
        max_y = max(coord[1] for coord in all_coords) + 1
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    
    # 添加标题和标签
    ax.set_title('路径规划可视化')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    
    # 显示图形
    plt.tight_layout()
    plt.show()

# 示例数据
input_example = {
    "all_vehicles": [
        [(0,0),(0,2),(2,2),(2,0)],   # 车0
        [(4,0),(4,2),(6,2),(6,0)],   # 车1 
        [(0,4),(0,6),(2,6),(2,4)],   # 车2
        [(4,4),(4,6),(6,6),(6,4)],   # 车3
        [(2,2),(2,4),(4,4),(4,2)]    # 车4
    ],
    "obstacle": [
        [(8,8),(8,10),(10,10),(10,8)]
    ],
    "destination": [(15,15),(15,17),(17,17),(17,15)],
    "path": [
        [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (0.0, 3.0), (-1.0, 3.0), (-1.0, 4.0), (-1.0, 5.0), (-1.0, 6.0), (-1.0, 7.0), (-1.0, 8.0), (-1.0, 9.0), (-1.0, 10.0), (-1.0, 11.0), (-1.0, 12.0), (-1.0, 13.0), (-1.0, 14.0), (0.0, 14.0), (1.0, 14.0), (2.0, 14.0), (3.0, 14.0), (4.0, 14.0), (5.0, 14.0), (6.0, 14.0), (7.0, 14.0), (8.0, 14.0), (9.0, 14.0), (10.0, 14.0), (11.0, 14.0), (12.0, 14.0), (13.0, 14.0), (14.0, 14.0), (15.0, 14.0), (16.0, 14.0), (17.0, 14.0), (18.0, 14.0), (18.0, 15.0), (18.0, 16.0), (19.0, 16.0), (20.0, 16.0), (21.0, 16.0), (22.0, 16.0)],
        [(1.0, 5.0), (1.0, 6.0), (1.0, 7.0), (1.0, 8.0), (1.0, 9.0), (1.0, 10.0), (1.0, 11.0), (1.0, 12.0), (1.0, 13.0), (1.0, 14.0), (1.0, 15.0), (1.0, 16.0), (1.0, 17.0), (1.0, 18.0), (1.0, 19.0), (1.0, 20.0), (1.0, 21.0), (2.0, 21.0), (3.0, 21.0), (4.0, 21.0), (5.0, 21.0), (6.0, 21.0), (7.0, 21.0), (8.0, 21.0), (9.0, 21.0), (10.0, 21.0), (11.0, 21.0), (12.0, 21.0), (13.0, 21.0)],
          [(3.0, 3.0), (3.0, 4.0), (3.0, 5.0), (3.0, 6.0), (3.0, 7.0), (4.0, 7.0), (5.0, 7.0), (6.0, 7.0), (7.0, 7.0), (8.0, 7.0), (9.0, 7.0), (10.0, 7.0), (11.0, 7.0), (11.0, 8.0), (11.0, 9.0), (11.0, 10.0), (12.0, 10.0), (13.0, 10.0), (13.0, 11.0)]]
}

# 运行可视化函数
visualize_path(input_example)    