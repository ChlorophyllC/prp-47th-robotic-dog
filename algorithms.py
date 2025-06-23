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

def a_star_path_planning(input_dict: Dict) -> List[Tuple[int, int]]:
    """
    A* path planning algorithm to navigate from a vehicle to a destination, avoiding obstacles.
    
    Args:
        input_dict: A dictionary containing:
            - 'vehicle': List of four corner points (x, y) representing the vehicle's rectangle.
            - 'obstacle': List of obstacles, each obstacle is a list of four corner points (x, y).
            - 'destination': A tuple (x, y) representing the destination point.
    
    Returns:
        A list of (x, y) tuples representing the path from vehicle's center to destination.
    """
    
    def is_collision(point: Tuple[int, int], buffer: int = 1) -> bool:
        """Check if a point is inside any obstacle or too close to it."""
        for obs in input_dict['obstacle']:
            if point_to_rect_distance(point, obs) < buffer:
                return True
        return False
    
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Get start and goal positions
    start = get_rect_center(input_dict['vehicle'])
    goal = input_dict['destination']
    
    # Define possible movements (up, down, left, right)
    movements = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    # Initialize data structures
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # Reverse to get from start to goal
        
        closed_set.add(current)
        
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Skip if out of bounds (we need some bounds, let's assume a large area)
            if abs(neighbor[0]) > 1000 or abs(neighbor[1]) > 1000:
                continue
                
            # Skip if in closed set
            if neighbor in closed_set:
                continue
                
            # Skip if collision
            if is_collision(neighbor):
                continue
                
            tentative_g = g_score[current] + 1  # All moves cost 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # No path found

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
