import math
import heapq
from typing import List, Tuple, Dict, Optional

class PathPlanner:
    def __init__(self, vehicles: List[List[Tuple[float, float]]], 
                 obstacles: List[List[Tuple[float, float]]], 
                 destinations: List[List[Tuple[float, float]]]):
        """
        初始化路径规划器
        
        Args:
            vehicles: 所有车辆的矩形定义，每个车辆由4个角点组成
            obstacles: 所有障碍物的矩形定义，每个障碍物由4个角点组成  
            destinations: 所有目的地的矩形定义，每个目的地由4个角点组成
        """
        self.vehicles = vehicles
        self.obstacles = obstacles
        self.destinations = destinations

    @staticmethod
    def get_rect_center(rect: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compute the center of a rectangle."""
        xs = [p[0] for p in rect]
        ys = [p[1] for p in rect]
        return (sum(xs) / 4, sum(ys) / 4)

    @staticmethod
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

    @staticmethod
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

    def get_all_obstacles(self, current_vehicle_index: int = None, target_destination_index: int = None) -> List[List[Tuple[float, float]]]:
        """
        获取所有障碍物，包括静态障碍物、其他车辆和其他目的地
        
        Args:
            current_vehicle_index: 当前车辆索引，如果指定则排除该车辆
            target_destination_index: 目标目的地索引，如果指定则排除该目的地
        """
        all_obstacles = []
        
        # 添加静态障碍物
        all_obstacles.extend(self.obstacles)
        
        # 添加其他车辆作为障碍物
        if current_vehicle_index is not None:
            all_obstacles.extend([vehicle for i, vehicle in enumerate(self.vehicles) if i != current_vehicle_index])
        else:
            all_obstacles.extend(self.vehicles)
        
        # 添加其他目的地作为障碍物
        if target_destination_index is not None:
            all_obstacles.extend([dest for i, dest in enumerate(self.destinations) if i != target_destination_index])
        else:
            all_obstacles.extend(self.destinations)
        
        return all_obstacles

    def is_valid_position(self, pos: Tuple[float, float], vehicle_rect: List[Tuple[float, float]], 
                         dest_center: Tuple[float, float], min_distance: float, buffer: float,
                         current_vehicle_index: int = None, target_destination_index: int = None) -> bool:
        """Check if a vehicle position is valid (no collisions and safe distance from destination)."""
        # Check distance to destination
        if math.hypot(pos[0] - dest_center[0], pos[1] - dest_center[1]) < min_distance:
            return False
        
        # Check collision with all obstacles
        all_obstacles = self.get_all_obstacles(current_vehicle_index, target_destination_index)
        for obs in all_obstacles:
            if self.rects_overlap(vehicle_rect, obs):
                return False
            if self.point_to_rect_distance(pos, obs) < buffer:
                return False
        return True

    def a_star_path_planning(self, current_vehicle_index: int, destination_index: int, max_iter: int = 10000, off_draw_mode = False, dest_point: Tuple[float, float] = None) -> List[Tuple[float, float]]:
        """
        A* path planning algorithm to navigate from a vehicle to a destination, avoiding obstacles.
        
        Args:
            current_vehicle_index: 当前车辆的索引
            destination_index: 目标目的地的索引
            max_iter: 最大迭代次数
        
        Returns:
            A list of (x, y) tuples representing the path from vehicle's center to destination.
        """
        def is_collision(point: Tuple[float, float], buffer: float = 1.0) -> bool:
            """Check collision with all obstacles"""
            all_obstacles = self.get_all_obstacles(current_vehicle_index, destination_index)
            
            for obs in all_obstacles:
                if self.point_to_rect_distance(point, obs) < buffer:
                    return True
            return False

        def heuristic(a, b):
            """Manhattan distance compatible with floats"""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Get start and goal positions
        current_vehicle = self.vehicles[current_vehicle_index]
        vehicle_width = int(max(p[0] for p in current_vehicle) - min(p[0] for p in current_vehicle))
        vehicle_height = int(max(p[1] for p in current_vehicle) - min(p[1] for p in current_vehicle))
        vehicle_max_size = max(vehicle_width, vehicle_height)
        start = self.get_rect_center(current_vehicle)
        
        if dest_point is not None:
            goal = dest_point
        else:
            target_destination = self.destinations[destination_index]
            goal = self.get_rect_center(target_destination)
        
        # Calculate reasonable boundaries
        all_points = []
        # Add all vehicle corner points
        for vehicle in self.vehicles:
            all_points.extend(vehicle)
        # Add all destination corner points
        for dest in self.destinations:
            all_points.extend(dest)
        # Add all obstacle corner points
        for obs in self.obstacles:
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

            # Floating point comparison needs tolerance
            if math.isclose(current[0], goal[0], abs_tol=0.5) and math.isclose(current[1], goal[1], abs_tol=0.5):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                if not off_draw_mode:
                    return [path[::-1]]
                return path[::-1]

            closed_set.add(current)
            
            for dx, dy in movements:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Stricter boundary check
                if not (min_x <= neighbor[0] <= max_x and min_y <= neighbor[1] <= max_y):
                    continue
                    
                if neighbor in closed_set:
                    continue
                    
                if is_collision(neighbor, vehicle_max_size):
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f"Warning: A* reached max iterations ({max_iter}) or no path found")
        return []

    def encirclement_algorithm(self, destination_index: int, vehicle_indices: List[int] = None, 
                             min_distance: float = 3.0, buffer: float = 1.0) -> List[Tuple[float, float]]:
        """
        Helper function: Evenly distribute vehicles around destination.
        
        Args:
            destination_index: 目标目的地的索引
            vehicle_indices: 参与包围的车辆索引列表，如果为None则使用所有车辆
            min_distance: 最小距离
            buffer: 安全缓冲区
        """
        if vehicle_indices is None:
            vehicle_indices = list(range(len(self.vehicles)))

        dest_rect = self.destinations[destination_index]
        dest_center = self.get_rect_center(dest_rect)
            # 计算目标矩形尺寸
        xs = [p[0] for p in dest_rect]
        ys = [p[1] for p in dest_rect]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # 基础半轴长度（椭圆的形式）
        a = width / 2 + min_distance + buffer  # x 方向半轴
        b = height / 2 + min_distance + buffer  # y 方向半轴
        
        num_vehicles = len(vehicle_indices)
        angles = [i * (2 * math.pi / num_vehicles) for i in range(num_vehicles)]
        
        vehicle_positions = []
        for angle in angles:
            # 使用椭圆公式获取包围坐标
            x = dest_center[0] + a * math.cos(angle)
            y = dest_center[1] + b * math.sin(angle)
            vehicle_positions.append((x, y))
        
        adjusted_positions = []
        for i, pos in enumerate(vehicle_positions):
            vehicle_idx = vehicle_indices[i]
            vehicle_rect = self.vehicles[vehicle_idx]
            dx = pos[0] - self.get_rect_center(vehicle_rect)[0]
            dy = pos[1] - self.get_rect_center(vehicle_rect)[1]
            shifted_rect = [(p[0] + dx, p[1] + dy) for p in vehicle_rect]

            # 尝试多个半轴微调以避免碰撞
            valid = False
            for delta in range(0, 5):  # 最多尝试 +4 的额外半径
                scale = 1.0 + 0.1 * delta
                test_x = dest_center[0] + scale * a * math.cos(angles[i])
                test_y = dest_center[1] + scale * b * math.sin(angles[i])
                dx = test_x - self.get_rect_center(vehicle_rect)[0]
                dy = test_y - self.get_rect_center(vehicle_rect)[1]
                shifted_rect = [(p[0] + dx, p[1] + dy) for p in vehicle_rect]

                if self.is_valid_position((test_x, test_y), shifted_rect, dest_center, 
                                        min_distance, buffer, vehicle_idx, destination_index):
                    adjusted_positions.append((test_x, test_y))
                    valid = True
                    break
            
            if not valid:
                # fallback
                adjusted_positions.append(pos)
        
        return adjusted_positions

    def expulsion_algorithm(self, direction: str, destination_index: int, vehicle_indices: List[int] = None,
                          min_distance: float = 3.0, buffer: float = 1.0) -> List[Tuple[float, float]]:
        """
        Push the destination in a specified direction (W/A/S/D) while semi-encircling it with vehicles.
        
        Args:
            direction: 'W' (up), 'A' (left), 'S' (down), 'D' (right). If None, push to nearest boundary.
            destination_index: 目标目的地的索引
            vehicle_indices: 参与驱逐的车辆索引列表，如果为None则使用所有车辆
            min_distance: Minimum distance between vehicles and destination.
            buffer: Safety buffer to avoid collisions.
        
        Returns:
            List of (x, y) positions for each vehicle's center.
        """
        if vehicle_indices is None:
            vehicle_indices = list(range(len(self.vehicles)))
        
        # Step 1: Determine push direction (default: nearest boundary)
        dest_center = self.get_rect_center(self.destinations[destination_index])
        if not direction:
            # Find nearest boundary (simplified: push towards closest axis)
            if dest_center[0] < dest_center[1]:
                direction = 'A' if dest_center[0] < 0 else 'D'
            else:
                direction = 'S' if dest_center[1] < 0 else 'W'
        
        # Step 2: Simulate vehicle_count + 1 vehicles for full encirclement
        extended_vehicle_indices = vehicle_indices + [vehicle_indices[0]]  # 复制第一个车辆
        full_positions = self.encirclement_algorithm(destination_index, extended_vehicle_indices, min_distance, buffer)
        
        # Step 3: Remove the vehicle in the push direction to form a semi-circle
        if len(full_positions) != len(vehicle_indices) + 1:
            print("Warning: Could not generate enough positions. Returning default encirclement.")
            return self.encirclement_algorithm(destination_index, vehicle_indices, min_distance, buffer)
        
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

    def sweep(self, current_vehicle_index: int, destination_index: int) -> List[Tuple[float, float]]:
        """
        基于边界点生成和A*路径规划的清扫算法
        Args:
            current_vehicle_index: 当前车辆索引
            destination_index: 目标目的地索引

        输出：严格上下左右移动的路径[(x1,y1), (x2,y2), ...]
        """
        # 1. 获取当前车辆参数
        current_vehicle = self.vehicles[current_vehicle_index]
        target_destination = self.destinations[destination_index]
        vehicle_width = int(max(p[0] for p in current_vehicle) - min(p[0] for p in current_vehicle))
        vehicle_height = int(max(p[1] for p in current_vehicle) - min(p[1] for p in current_vehicle))
        
        # 2. 计算目标区域边界（整数坐标）
        min_x = int(min(p[0] for p in target_destination))
        max_x = int(max(p[0] for p in target_destination))
        min_y = int(min(p[1] for p in target_destination))
        max_y = int(max(p[1] for p in target_destination))
        
        # 3. 生成入口关键点（考虑避障）
        def find_closest_corner(dest, vehicle, obstacles):
            # 计算车辆的中心点
            vehicle_center = self.get_rect_center(vehicle)
            
            # 检查每个角点
            valid_corners = []
            for corner in dest:
                # 检查点是否在任何障碍物内
                if any(self.rects_overlap([corner], obs) for obs in obstacles):
                    continue
                valid_corners.append(corner)
            
            if not valid_corners:
                return []
            # 找到距离车辆中心最近的角点
            closest_corner = min(valid_corners, key=lambda c: math.hypot(c[0] - vehicle_center[0], c[1] - vehicle_center[1]))
            return closest_corner
        
        all_obstacles = self.get_all_obstacles(current_vehicle_index, destination_index)
        closest_corner = find_closest_corner(target_destination, current_vehicle, all_obstacles)

        # 生成前往最近角点的路径
        path = self.a_star_path_planning(current_vehicle_index, destination_index, off_draw_mode=True, dest_point=closest_corner)
        
        # 从该角点开始生成 Z 字形路径
        def generate_zigzag_path(start, dest, vehicle, step_size=None):
            path = []
            x, y = start

            # 根据车辆尺寸设置步长
            vehicle_width = int(max(p[0] for p in vehicle) - min(p[0] for p in vehicle))
            vehicle_height = int(max(p[1] for p in vehicle) - min(p[1] for p in vehicle))
            if step_size is None:
                step_size = min(vehicle_width, vehicle_height)
            
            # 边界框
            xs = [p[0] for p in dest]
            ys = [p[1] for p in dest]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # 起点位置判断（增加容差）
            if math.isclose(x, min_x, abs_tol=1.0) and math.isclose(y, min_y, abs_tol=1.0):
                x_dir, y_dir = 1, 1
            elif math.isclose(x, max_x, abs_tol=1.0) and math.isclose(y, min_y, abs_tol=1.0):
                x_dir, y_dir = -1, 1
            elif math.isclose(x, min_x, abs_tol=1.0) and math.isclose(y, max_y, abs_tol=1.0):
                x_dir, y_dir = 1, -1
            elif math.isclose(x, max_x, abs_tol=1.0) and math.isclose(y, max_y, abs_tol=1.0):
                x_dir, y_dir = -1, -1
            else:
                x_dir, y_dir = 1, 1  # 默认从左下角向右上角

            # 是否水平优先
            horizontal_priority = (max_x - min_x) >= (max_y - min_y)

            if horizontal_priority:
                y_line = y
                while (y_dir > 0 and y_line <= max_y) or (y_dir < 0 and y_line >= min_y):
                    # 当前行扫描
                    x_pos = min_x if x_dir > 0 else max_x
                    line = []
                    while (x_dir > 0 and x_pos <= max_x) or (x_dir < 0 and x_pos >= min_x):
                        line.append((x_pos, y_line))
                        x_pos += x_dir * step_size
                    path.extend(line)
                    
                    y_line += y_dir * vehicle_height
                    x_dir *= -1
            else:
                x_col = x
                while (x_dir > 0 and x_col <= max_x) or (x_dir < 0 and x_col >= min_x):
                    # 当前列扫描
                    y_pos = min_y if y_dir > 0 else max_y
                    col = []
                    while (y_dir > 0 and y_pos <= max_y) or (y_dir < 0 and y_pos >= min_y):
                        col.append((x_col, y_pos))
                        y_pos += y_dir * step_size
                    path.extend(col)
                    
                    x_col += x_dir * vehicle_width
                    y_dir *= -1

            return path

        path += generate_zigzag_path(start=closest_corner, dest=target_destination, vehicle=current_vehicle)

        return [path]

    @staticmethod
    def push_destination(direction, destination_rect, step_size=0.5):
        """Push destination rectangle according to direction"""
        dx, dy = {
            'W': (0, step_size),    # Up
            'A': (-step_size, 0),   # Left
            'S': (0, -step_size),   # Down
            'D': (step_size, 0)     # Right
        }[direction]
        
        return [(x+dx, y+dy) for x,y in destination_rect]

    def encirclement_implement(self, destination_index: int, selected_vehicle_indices: List[int]) -> List[List[Tuple[float, float]]]:
        """
        Improved cooperative encirclement algorithm that can select specific vehicles to participate
        
        Args:
            destination_index: 目标目的地的索引
            selected_vehicle_indices: List of indices of vehicles to participate in encirclement
        """
        # Calculate encirclement positions for selected vehicles
        target_positions = self.encirclement_algorithm(destination_index, selected_vehicle_indices)
        
        # Plan path for each selected vehicle
        all_paths = []
        for idx, vehicle_idx in enumerate(selected_vehicle_indices):
            # 创建临时目的地用于A*规划
            temp_dest_index = len(self.destinations)
            target_pos = target_positions[idx]
            self.destinations.append([target_pos, target_pos, target_pos, target_pos])  # 临时添加目标位置作为目的地
            
            path = self.a_star_path_planning(vehicle_idx, temp_dest_index, off_draw_mode=True)
            all_paths.append(path)
            
            # 移除临时目的地
            self.destinations.pop()
        
        return all_paths
    
# # # 初始化
# vehicles=[[[41,23],[41,20],[45,20],[45,23]],[[43,34],[43,30],[47,30],[47,34]]]
# obstacles=[[[17,38],[17,30],[28,30],[28,38]]]
# destinations=[[[17,19],[17,7],[29,7],[29,19]]]

# planner = PathPlanner(vehicles, obstacles, destinations)

# # 使用A*规划从车辆0到目的地1的路径
# path = planner.a_star_path_planning(current_vehicle_index=0, destination_index=0)
# # 车辆0和2包围目的地0
# encirclement_path = planner.encirclement_implement(destination_index=0, selected_vehicle_indices=[0, 1])
# # 车辆0 清扫目的地0
# path = planner.sweep(current_vehicle_index=0, destination_index=0)
# print("A* Path:", path)
# print("Encirclement Path:", encirclement_path)