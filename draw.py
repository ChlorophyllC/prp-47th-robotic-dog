import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_polygon_patch(coords, facecolor, edgecolor='black', alpha=0.7):
    """创建多边形补丁"""
    polygon = patches.Polygon(coords, facecolor=facecolor, edgecolor=edgecolor, 
                             alpha=alpha, linewidth=2)
    return polygon

def visualize_static_path():
    """静态路径可视化"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制车辆
    for vehicle in input_example["all_vehicles"]:
        vehicle_patch = create_polygon_patch(vehicle, 'lightgreen', 'darkgreen')
        ax.add_patch(vehicle_patch)
    
    # 绘制障碍物
    for obstacle in input_example["obstacle"]:
        obstacle_patch = create_polygon_patch(obstacle, 'lightcoral', 'darkred')
        ax.add_patch(obstacle_patch)
    
    # 绘制目的地
    for dest in input_example["destination"]:
        dest_patch = create_polygon_patch(dest, 'lightblue', 'darkblue')
        ax.add_patch(dest_patch)
    
    # 绘制路径
    for path in input_example["path"]:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        
        # 绘制路径线
        ax.plot(path_x, path_y, 'orange', linewidth=2, linestyle='--', 
                label='规划路径', marker='o', markersize=4, alpha=0.8)
    
        # 标注路径点序号
        for i, (x, y) in enumerate(path):
            ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # 标注起点和终点
        start_point = path[0]
        end_point = path[-1]
        ax.plot(start_point[0], start_point[1], 'go', markersize=12, label='起点')
        ax.plot(end_point[0], end_point[1], 'ro', markersize=12, label='终点')
    
    # 设置图形属性
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 20)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.set_title('路径规划可视化', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='车辆'),
        patches.Patch(facecolor='lightcoral', edgecolor='darkred', label='障碍物'),
        patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='目的地'),
        plt.Line2D([0], [0], color='orange', linewidth=3, linestyle='--', label='规划路径'),
        plt.Line2D([0], [0], marker='o', color='green', markersize=8, label='起点', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='red', markersize=8, label='终点', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.show()

def visualize_animated_path():
    """动画路径可视化"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制静态元素
    # 车辆
    for vehicle in input_example["all_vehicles"]:
        vehicle_patch = create_polygon_patch(vehicle, 'lightgreen', 'darkgreen')
        ax.add_patch(vehicle_patch)
    
    # 障碍物
    for obstacle in input_example["obstacle"]:
        obstacle_patch = create_polygon_patch(obstacle, 'lightcoral', 'darkred')
        ax.add_patch(obstacle_patch)
    
    # 目的地
    for dest in input_example["destination"]:
        dest_patch = create_polygon_patch(dest, 'lightblue', 'darkblue')
        ax.add_patch(dest_patch)
    
    # 路径数据
    path = input_example["path"][0]
    path_x = [point[0] for point in path]
    path_y = [point[1] for point in path]
    
    # 绘制完整路径（浅色）
    ax.plot(path_x, path_y, 'orange', linewidth=2, linestyle='--', alpha=0.3)
    
    # 初始化动画元素
    line, = ax.plot([], [], 'orange', linewidth=4, marker='o', markersize=6)
    current_pos, = ax.plot([], [], 'purple', marker='o', markersize=15, alpha=0.8)
    
    # 文本显示当前步骤
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       verticalalignment='top')
    
    def animate(frame):
        # 更新已走过的路径
        if frame > 0:
            line.set_data(path_x[:frame+1], path_y[:frame+1])
        
        # 更新当前位置
        if frame < len(path):
            current_pos.set_data([path_x[frame]], [path_y[frame]])
            step_text.set_text(f'步骤: {frame+1}/{len(path)}\n'
                              f'当前位置: ({path_x[frame]:.1f}, {path_y[frame]:.1f})')
        
        return line, current_pos, step_text
    
    # 设置图形属性
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 20)
    ax.set_xlabel('X坐标', fontsize=12)
    ax.set_ylabel('Y坐标', fontsize=12)
    ax.set_title('路径规划动画演示', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加图例
    legend_elements = [
        patches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='车辆'),
        patches.Patch(facecolor='lightcoral', edgecolor='darkred', label='障碍物'),
        patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='目的地'),
        plt.Line2D([0], [0], color='orange', linewidth=3, label='规划路径'),
        plt.Line2D([0], [0], marker='o', color='purple', markersize=10, label='当前位置', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 创建动画
    anim = FuncAnimation(fig, animate, frames=len(path), interval=500, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.savefig('path_animation.png', dpi=300)  # 保存静态图像
    
    return anim

def print_path_info():
    """打印路径信息"""
    path = input_example["path"][0]
    print("="*50)
    print("路径规划详细信息")
    print("="*50)
    print(f"起点: ({path[0][0]}, {path[0][1]})")
    print(f"终点: ({path[-1][0]}, {path[-1][1]})")
    print(f"总步数: {len(path)}")
    print("\n路径序列:")
    print("-"*30)
    
    for i, (x, y) in enumerate(path):
        direction = ""
        if i > 0:
            prev_x, prev_y = path[i-1]
            if x > prev_x:
                direction = "→ 东"
            elif x < prev_x:
                direction = "← 西"
            elif y > prev_y:
                direction = "↑ 北"
            elif y < prev_y:
                direction = "↓ 南"
        
        print(f"步骤 {i+1:2d}: ({x:4.1f}, {y:4.1f}) {direction}")
    
    # 计算路径长度
    total_distance = 0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        total_distance += distance
    
    print(f"\n总路径长度: {total_distance:.2f} 单位")
    print("="*50)

def main():
    """主函数"""
    print("路径规划可视化程序")
    print("1. 显示路径信息")
    print("2. 静态路径可视化")
    print("3. 动画路径可视化")
    
    # 显示路径信息
    print_path_info()
    
    # 静态可视化
    print("\n正在生成静态路径可视化...")
    visualize_static_path()
    
    # 询问是否显示动画
    choice = input("\n是否显示动画演示? (y/n): ").lower().strip()
    if choice == 'y' or choice == 'yes':
        print("正在生成动画路径可视化...")
        anim = visualize_animated_path()
        
        # 可选：保存动画为GIF
        save_choice = input("是否保存动画为GIF? (y/n): ").lower().strip()
        if save_choice == 'y' or save_choice == 'yes':
            print("正在保存动画... (这可能需要一些时间)")
            anim.save('path_animation.gif', writer='pillow', fps=2)
            print("动画已保存为 'path_animation.gif'")

# 示例数据
input_example = {
    "all_vehicles": [[[3, 18], [3, 16], [5, 16], [5, 18]],
    [[9, 18], [9, 17], [10, 17], [10, 18]],
    [[3, 8], [3, 9], [4, 9], [4, 8]],
    [[17, 15], [17, 14], [18, 14], [18, 15]]],
    "obstacle": 
        [[[8, 14], [8, 13], [10, 13], [10, 14]],
    [[11, 16], [11, 12], [12, 12], [12, 16]],
    [[3, 12], [3, 11], [5, 11], [5, 12]]]
    ,
    "destination": [
    [[11, 8], [11, 5], [13, 5], [13, 8]],
    [[3, 5], [3, 3], [5, 3], [5, 5]]
],
    "path": [
        [(4.0, 17.0), (4.0, 16.0), (4.0, 15.0), (4.0, 14.0), (4.0, 13.0), (5.0, 13.0), (6.0, 13.0), (6.0, 12.0), (6.0, 11.0), (6.0, 10.0), (6.0, 9.0), (7.0, 9.0), (8.0, 9.0), (9.0, 9.0), (10.0, 9.0), (11.0, 9.0), (12.0, 9.0), (13.0, 9.0), (14.0, 9.0), (14.0, 8.0), (14.0, 7.0), (15.0, 7.0), (16.0, 7.0)], [(3.5, 8.5), (3.5, 7.5), (4.5, 7.5), (5.5, 7.5), (6.5, 7.5), (7.5, 7.5), (7.5, 6.5)]
        ]
}

if __name__ == "__main__":
    main()


  