你**能且仅能**从以下三个函数中选择，使用时均需要**传入参数**，你的回答是形如[a_star_path_planning(input_dict = {"all_vehicles": [[(0,0),(0,2),(2,2),(2,0)]],"obstacle": [[(8,8),(8,10),(10,10),(10,8)]],"destination": [(15,15),(15,17),(17,17),(17,15)]}, current_vehicle_index = 0)] 的列表。
1. A* 路径规划算法
```python
def a_star_path_planning(
    input_dict: Dict, 
    current_vehicle_index: int = 0, 
    max_iter: int = 10000
) -> List[Tuple[float, float]]:
"""
基于A*算法计算指定车辆到目标点的无碰撞路径

输入参数:
    input_dict: 包含以下字段的字典
        - all_vehicles: 所有车辆的矩形区域顶点坐标列表（每个车辆由四个角点表示）
        - obstacle: 障碍物的矩形区域顶点坐标列表（每个障碍物由四个角点表示）
        - destination: 目标点坐标元组(x, y)
    current_vehicle_index: 待规划路径的车辆在all_vehicles中的索引

返回值:
    从车辆当前位置到目标点的路径点列表，格式为[(x1,y1), (x2,y2), ...]
"""
```
2. 协同包围算法
```python
def encirclement_implement(
    input_dict: Dict, 
    selected_vehicle_indices: List[int]
) -> List[List[Tuple[float, float]]]:
"""
指挥多车辆协同包围目标区域的路径规划算法

输入参数:
    input_dict: 包含以下字段的字典
        - all_vehicles: 所有车辆的矩形区域顶点坐标列表（每个车辆由四个角点表示）
        - obstacle: 障碍物的矩形区域顶点坐标列表（每个障碍物由四个角点表示）
        - destination: 目标区域的矩形顶点坐标列表（由四个角点表示）
    selected_vehicle_indices: 参与包围任务的车辆索引列表

返回值:
    每个参与车辆的路径点列表，格式为：
    [
        [(x1,y1), (x2,y2), ...],
        [(x1,y1), (x2,y2), ...],
        ...
    ]
"""
```