import math
import heapq
from typing import List, Tuple, Dict, Optional

def get_rect_center(rect: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute the center of a rectangle."""
    xs = [p[0] for p in rect]
    ys = [p[1] for p in rect]
    return (sum(xs) / 4, sum(ys) / 4)

def rects_overlap(rect1: List[Tuple[float, float]], rect2: List[Tuple[float, float]]) -> bool:
    """Check if two rectangles overlap (AABB collision detection)."""
    rect1_xs = [p[0] for p in rect1]
    rect1_ys = [p[1] for p in rect1]
    rect2_xs = [p[0] for p in rect2]
    rect2_ys = [p[1] for p in rect2]
    
    return (
        max(rect1_xs) >= min(rect2_xs) and
        min(rect1_xs) <= max(rect2_xs) and
        max(rect1_ys) >= min(rect2_ys) and
        min(rect1_ys) <= max(rect2_ys)
    )

def point_to_rect_distance(point: Tuple[float, float], rect: List[Tuple[float, float]]) -> float:
    """Compute minimal distance from a point to a rectangle."""
    px, py = point
    min_dist = float('inf')
    
    for i in range(4):
        x1, y1 = rect[i]
        x2, y2 = rect[(i + 1) % 4]
        
        edge_len = math.hypot(x2 - x1, y2 - y1)
        if edge_len == 0:
            continue
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (edge_len ** 2)))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        dist = math.hypot(px - proj_x, py - proj_y)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def is_valid_position(pos: Tuple[float, float], vehicle_rect: List[Tuple[float, float]], obstacles: List[List[Tuple[float, float]]], dest_center: Tuple[float, float], min_distance: float, buffer: float) -> bool:
    """Check if a vehicle position is valid (no collisions and safe distance from destination)."""
    # Check distance to destination
    if math.hypot(pos[0] - dest_center[0], pos[1] - dest_center[1]) < min_distance:
        return False
    
    # Check collision with obstacles
    for obs in obstacles:
        if rects_overlap(vehicle_rect, obs):
            return False
        if point_to_rect_distance(pos, obs) < buffer:
            return False
    return True

def a_star_path_planning(input_dict: Dict, current_vehicle_index: int, max_iter: int = 10000) -> List[Tuple[int, int]]:
    """
    A* path planning algorithm to navigate from a vehicle to a destination, avoiding obstacles.
    
    Args:
        input_dict: A dictionary containing:
            - 'all_vehicles': List of four corner points (x, y) representing all vehicles' rectangles.
            - 'obstacle': List of obstacles, each obstacle is a list of four corner points (x, y).
            - 'destination': A tuple (x, y) representing the destination point.
        - current_vehicle_index: 当前车辆的索引    
    
    Returns:
        A list of (x, y) tuples representing the path from vehicle's center to destination.
    """

    def get_other_vehicles():
        """获取其他车辆的障碍物"""
        return [
            vehicle for i, vehicle in enumerate(input_dict['all_vehicles'])
            if i != current_vehicle_index
        ]
    
    def is_collision(point: Tuple[float, float], buffer: float = 1.0) -> bool:
        """检查与静态障碍物和其他车辆的碰撞"""
        # 合并障碍物
        all_obstacles = input_dict['obstacle'] + get_other_vehicles()
        
        for obs in all_obstacles:
            if point_to_rect_distance(point, obs) < buffer:
                return True
        return False

    def heuristic(a, b):
        """兼容float的曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # 统一坐标类型
    current_vehicle = input_dict['all_vehicles'][current_vehicle_index]
    start = get_rect_center(current_vehicle)
    goal = (float(input_dict['destination'][0]), float(input_dict['destination'][1]))
    
    # 计算合理边界
    all_points = []
    # 添加所有车辆的角点
    for vehicle in input_dict['all_vehicles']:
        all_points.extend(vehicle)
    # 添加目标点
    all_points.append(goal)
    # 添加所有障碍物的角点
    for obs in input_dict['obstacle']:
        all_points.extend(obs)
    min_x = min(p[0] for p in all_points) - 5
    max_x = max(p[0] for p in all_points) + 5
    min_y = min(p[1] for p in all_points) - 5
    max_y = max(p[1] for p in all_points) + 5

    movements = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    iter_count = 0

    while open_set and iter_count < max_iter:
        iter_count += 1
        current_f, current = heapq.heappop(open_set)

        # 浮点数比较需要容差
        if math.isclose(current[0], goal[0], abs_tol=0.5) and math.isclose(current[1], goal[1], abs_tol=0.5):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        closed_set.add(current)
        
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # 更严格的边界检查
            if not (min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y):
                continue
                
            if neighbor in closed_set:
                continue
                
            if is_collision(neighbor):
                continue
                
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print(f"Warning: A* reached max iterations ({max_iter}) or no path found")
    return []

def encirclement_algorithm(input_dict: Dict, min_distance: float = 3.0, buffer: float = 1.0) -> List[Tuple[float, float]]:
    """Helper function: Evenly distribute vehicles around destination."""
    dest_center = get_rect_center(input_dict['destination'])
    num_vehicles = len(input_dict['vehicle'])
    radius = max(min_distance * 1.5, num_vehicles * 2.0)
    angles = [i * (2 * math.pi / num_vehicles) for i in range(num_vehicles)]
    
    vehicle_positions = []
    for angle in angles:
        x = dest_center[0] + radius * math.cos(angle)
        y = dest_center[1] + radius * math.sin(angle)
        vehicle_positions.append((x, y))
    
    adjusted_positions = []
    for i, pos in enumerate(vehicle_positions):
        vehicle_rect = input_dict['vehicle'][i] if i < len(input_dict['vehicle']) else input_dict['vehicle'][0]
        dx = pos[0] - get_rect_center(vehicle_rect)[0]
        dy = pos[1] - get_rect_center(vehicle_rect)[1]
        shifted_rect = [(p[0] + dx, p[1] + dy) for p in vehicle_rect]
        
        if is_valid_position(pos, shifted_rect, input_dict['obstacle'], dest_center, min_distance, buffer):
            adjusted_positions.append(pos)
        else:
            new_radius = radius + 1.0
            new_x = dest_center[0] + new_radius * math.cos(angle)
            new_y = dest_center[1] + new_radius * math.sin(angle)
            new_pos = (new_x, new_y)
            new_dx = new_pos[0] - get_rect_center(vehicle_rect)[0]
            new_dy = new_pos[1] - get_rect_center(vehicle_rect)[1]
            new_shifted_rect = [(p[0] + new_dx, p[1] + new_dy) for p in vehicle_rect]
            
            if is_valid_position(new_pos, new_shifted_rect, input_dict['obstacle'], dest_center, min_distance, buffer):
                adjusted_positions.append(new_pos)
            else:
                for delta_angle in [0.1, -0.1, 0.2, -0.2]:
                    adjusted_angle = angle + delta_angle
                    adjusted_x = dest_center[0] + radius * math.cos(adjusted_angle)
                    adjusted_y = dest_center[1] + radius * math.sin(adjusted_angle)
                    adjusted_pos = (adjusted_x, adjusted_y)
                    adjusted_dx = adjusted_pos[0] - get_rect_center(vehicle_rect)[0]
                    adjusted_dy = adjusted_pos[1] - get_rect_center(vehicle_rect)[1]
                    adjusted_shifted_rect = [(p[0] + adjusted_dx, p[1] + adjusted_dy) for p in vehicle_rect]
                    
                    if is_valid_position(adjusted_pos, adjusted_shifted_rect, input_dict['obstacle'], dest_center, min_distance, buffer):
                        adjusted_positions.append(adjusted_pos)
                        break
                else:
                    continue
    return adjusted_positions

def expulsion_algorithm(direction: str, input_dict: Dict, min_distance: float = 3.0, buffer: float = 1.0) -> List[Tuple[float, float]]:
    """
    Push the destination in a specified direction (W/A/S/D) while semi-encircling it with vehicles.
    
    Args:
        direction: 'W' (up), 'A' (left), 'S' (down), 'D' (right). If None, push to nearest boundary.
        input_dict: Contains 'vehicle', 'obstacle', 'destination' (each defined by 4 corner points).
        min_distance: Minimum distance between vehicles and destination.
        buffer: Safety buffer to avoid collisions.
    
    Returns:
        List of (x, y) positions for each vehicle's center.
    """
    # Step 1: Determine push direction (default: nearest boundary)
    dest_center = get_rect_center(input_dict['destination'])
    if not direction:
        # Find nearest boundary (simplified: push towards closest axis)
        if dest_center[0] < dest_center[1]:
            direction = 'A' if dest_center[0] < 0 else 'D'
        else:
            direction = 'S' if dest_center[1] < 0 else 'W'
    
    # Step 2: Simulate vehicle_count + 1 vehicles for full encirclement
    fake_vehicle = input_dict['vehicle'][0]  # Use first vehicle as template
    fake_input_dict = {
        'vehicle': input_dict['vehicle'] + [fake_vehicle],
        'obstacle': input_dict['obstacle'],
        'destination': input_dict['destination']
    }
    full_positions = encirclement_algorithm(fake_input_dict, min_distance, buffer)
    
    # Step 3: Remove the vehicle in the push direction to form a semi-circle
    if len(full_positions) != len(input_dict['vehicle']) + 1:
        print("Warning: Could not generate enough positions. Returning default encirclement.")
        return encirclement_algorithm(input_dict, min_distance, buffer)
    
    # Find the vehicle to remove (opposite to push direction)
    angle_to_remove = {
        'W': math.pi / 2,    # Up (remove bottom)
        'A': math.pi,        # Left (remove right)
        'S': 3 * math.pi / 2, # Down (remove top)
        'D': 0               # Right (remove left)
    }[direction]
    
    # Find the position closest to the opposite direction
    def angle_diff(a1, a2):
        return min((a1 - a2) % (2 * math.pi), (a2 - a1) % (2 * math.pi))
    
    angles = [math.atan2(pos[1] - dest_center[1], pos[0] - dest_center[0]) % (2 * math.pi) for pos in full_positions]
    idx_to_remove = min(range(len(angles)), key=lambda i: angle_diff(angles[i], angle_to_remove))
    
    semi_positions = [pos for i, pos in enumerate(full_positions) if i != idx_to_remove]
    
    return semi_positions

def push_destination(direction, destination_rect, step_size=0.5):
    """根据方向推动目标矩形"""
    dx, dy = {
        'W': (0, step_size),    # 上
        'A': (-step_size, 0),   # 左
        'S': (0, -step_size),   # 下
        'D': (step_size, 0)     # 右
    }[direction]
    
    return [(x+dx, y+dy) for x,y in destination_rect]

def encirclement_implement(input_dict: Dict, selected_vehicle_indices: List[int]) -> List[List[Tuple[float, float]]]:
    """
    改进版协同包围算法，可选择部分车辆参与
    
    Args:
        input_dict: 包含所有车辆和障碍物信息
        selected_vehicle_indices: 指定参与包围的车辆索引列表
    """
    # 提取参与车辆
    selected_vehicles = [input_dict['all_vehicles'][i] for i in selected_vehicle_indices]
    
    # 计算包围位置时只考虑选定车辆
    partial_input = {
        'vehicle': selected_vehicles,
        'obstacle': input_dict['obstacle'],
        'destination': input_dict['destination']
    }
    target_positions = encirclement_algorithm(partial_input)
    
    # 为每辆选定车辆规划路径（考虑所有车辆作为障碍）
    all_paths = []
    for idx, vehicle_idx in enumerate(selected_vehicle_indices):
        planning_dict = {
            'all_vehicles': input_dict['all_vehicles'],  # 全部车辆信息
            'obstacle': input_dict['obstacle'] + [input_dict['destination']],           # 静态障碍物
            'destination': target_positions[idx]          # 该车的目标位置
        }
        path = a_star_path_planning(planning_dict, current_vehicle_index= vehicle_idx)
        all_paths.append(path)
    
    return all_paths

# 示例用法
if __name__ == "__main__":
    input_example = {"all_vehicles": [
        [(0,0),(0,2),(2,2),(2,0)],   # 车0
        [(4,0),(4,2),(6,2),(6,0)],   # 车1 
        [(0,4),(0,6),(2,6),(2,4)],   # 车2
        [(4,4),(4,6),(6,6),(6,4)],   # 车3
        [(2,2),(2,4),(4,4),(4,2)]    # 车4
    ],
    "obstacle": [
        [(8,8),(8,10),(10,10),(10,8)]
    ],
    "destination": [(15,15),(15,17),(17,17),(17,15)]
}
    
        # 选择车0、2、4执行包围
    selected_vehicles = [0, 2, 4]
    paths = encirclement_implement(input_example, selected_vehicles)

    # 结果可视化
    for i, path in zip(selected_vehicles, paths):
        print(f"Vehicle {i} path : {path}")

