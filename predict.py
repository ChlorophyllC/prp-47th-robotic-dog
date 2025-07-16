from ultralytics import YOLO
import os
import cv2
import numpy as np
import json
import time

# 获取当前文件的目录路径
path = os.path.dirname(os.path.abspath(__file__))

# 加载训练好的模型
model = YOLO('best.pt')

# 指定测试图片文件夹
class_names = ['Yellow', 'Red', 'Green', 'Blue', 'Vehicle', 'Ball']

def draw_boxes(image, results, class_names):
    """
    在图像上绘制检测框和标签
    :param image: 原始图像 (numpy array)
    :param results: YOLO检测结果
    :param class_names: 类别名称列表
    :return: 绘制后的图像
    """
    # 颜色表 (BGR格式)
    colors = [
        (0, 255, 255),  # Yellow
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (255, 255, 0),  # Vehicle
        (0, 165, 255)   # Ball (橙色)
    ]
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # 获取框坐标和类别
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            
            # 绘制矩形框
            color = colors[cls_id % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签和置信度
            label = f"{class_names[cls_id]}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return image

def convert_to_grid_coordinates(x1, y1, x2, y2, image_width, image_height, grid_width=20, grid_height=20):
    """
    将像素坐标转换为网格坐标
    """
    # 计算网格单元的大小
    cell_width = image_width / grid_width
    cell_height = image_height / grid_height
    
    # 转换为网格坐标
    grid_x1 = int(x1 / cell_width)
    grid_y1 = int((image_height - y1) / cell_height)
    grid_x2 = int(x2 / cell_width)
    grid_y2 = int((image_height - y2) / cell_height)

    # 确保坐标在有效范围内
    grid_x1 = max(0, min(grid_x1, grid_width - 1))
    grid_y1 = max(0, min(grid_y1, grid_height - 1))
    grid_x2 = max(0, min(grid_x2, grid_width - 1))
    grid_y2 = max(0, min(grid_y2, grid_height - 1))
    
    return grid_x1, grid_y1, grid_x2, grid_y2
from typing import List, Tuple

def convert_to_image_coordinates(grid_x: float, grid_y: float, 
                                 image_width: int = 1440, image_height: int = 1080, 
                                 grid_width: int = 144, grid_height: int = 108) -> Tuple[float, float]:
    """
    将网格坐标转换为图像像素坐标（中心点位置）
    
    :param grid_x: 网格坐标 x（可以是 float）
    :param grid_y: 网格坐标 y（可以是 float）
    :return: 像素坐标 (x, y)，以图像左上角为原点
    """
    cell_width = image_width / grid_width
    cell_height = image_height / grid_height

    # 网格中心点对应的图像坐标
    pixel_x = (grid_x + 0.5) * cell_width
    pixel_y = image_height - (grid_y + 0.5) * cell_height

    return pixel_x, pixel_y

def batch_convert_to_image_coordinates(
    grid_coords: List[Tuple[float, float]],
    image_width: int = 1440,
    image_height: int = 1080,
    grid_width: int = 144,
    grid_height: int = 108
) -> List[Tuple[float, float]]:
    """
    批量将网格坐标列表转换为图像像素坐标列表（中心点位置）
    
    :param grid_coords: 网格坐标列表，如 [(x1, y1), (x2, y2), ...]
    :param image_width: 图像宽度（像素）
    :param image_height: 图像高度（像素）
    :return: 图像坐标列表 [(px1, py1), (px2, py2), ...]
    """
    cell_width = image_width / grid_width
    cell_height = image_height / grid_height

    image_coords = []
    for grid_x, grid_y in grid_coords:
        pixel_x = (grid_x + 0.5) * cell_width
        pixel_y = image_height - (grid_y + 0.5) * cell_height
        image_coords.append((pixel_x, pixel_y))
    
    return image_coords

def batch_convert_to_grid_coordinates(
    image_coords: List[Tuple[float, float]],
    image_width: int = 1440,
    image_height: int = 1080,
    grid_width: int = 144,
    grid_height: int = 108
) -> List[Tuple[float, float]]:
    """
    批量将图像像素坐标列表转换为网格坐标列表
    
    :param image_coords: 图像坐标列表 [(px1, py1), (px2, py2), ...]
    :param image_width: 图像宽度（像素）
    :param image_height: 图像高度（像素）
    :param grid_width: 网格宽度（格子数）
    :param grid_height: 网格高度（格子数）
    :return: 网格坐标列表 [(x1, y1), (x2, y2), ...]
    """
    cell_width = image_width / grid_width
    cell_height = image_height / grid_height

    grid_coords = []
    for pixel_x, pixel_y in image_coords:
        # 计算网格x坐标（考虑中心点）
        grid_x = (pixel_x / cell_width) - 0.5
        # 计算网格y坐标（考虑Y轴反转和中心点）
        grid_y = ((image_height - pixel_y) / cell_height) - 0.5
        grid_coords.append((grid_x, grid_y))
    
    return grid_coords

def bbox_to_corners(x1, y1, x2, y2):
    """
    将边界框转换为四个角点坐标
    """
    return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

def detect_objects(path, show_results=False, verbose=True):
    """
    检测图像中的目标并返回原始检测结果（像素坐标）
    
    参数:
        path: 图像路径
        show_results: 是否显示检测结果图像
    
    返回:
        tuple: (检测结果字典, 图像尺寸), 字典格式为:
        {
            "all_vehicles": [bboxes...],
            "obstacle": [bboxes...],
            "destination": [bboxes...]
        }
        其中bboxes是(x1,y1,x2,y2)像素坐标
    """
    frame = cv2.imread(path)
    if frame is None:
        print(f"无法读取图像: {path}")
        return None
    
    # YOLO检测
    results = model.predict(frame, imgsz=640, conf=0.6, verbose=verbose)
    
    # 获取图像尺寸
    image_height, image_width = frame.shape[:2]

    # 初始化结果字典
    detection_results = {
        "all_vehicles": [],
        "obstacle": [],
        "destination": []
    }
    
    # 处理检测结果
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # 获取框坐标和类别
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # 根据类别分类
            class_name = class_names[cls_id]
            bbox = (x1, y1, x2, y2)
            if class_name == 'Red':
                detection_results["obstacle"].append(bbox)
            elif class_name == 'Green':
                detection_results["destination"].append(bbox)
            elif class_name == 'Vehicle':
                detection_results["all_vehicles"].append(bbox)
    
    if show_results:
        # 绘制检测框并显示
        annotated_frame = draw_boxes(frame.copy(), results, class_names)
        
        # 显示实时结果
        cv2.imshow("YOLOv11 Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detection_results, (image_width, image_height)

def save_detection_results(detection_data, save_dir="detection_results", verbose=True):
    """
    将检测结果转换为网格坐标并保存为JSON文件
    
    参数:
        detection_data: 包含检测结果和图像尺寸的元组 (detection_results, image_size)
        save_dir: 保存目录
    
    返回:
        str: 保存的文件路径
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if detection_data is None:
        print("无有效检测结果可保存")
        return None
    
    detection_results, (image_width, image_height) = detection_data
    
    # 转换为网格坐标的结果字典
    grid_results = {
        "all_vehicles": [],
        "obstacle": [],
        "destination": []
    }
    
    # 将每个bbox转换为网格坐标
    for category in detection_results:
        for bbox in detection_results[category]:
            x1, y1, x2, y2 = bbox
            # 转换为网格坐标
            grid_x1, grid_y1, grid_x2, grid_y2 = convert_to_grid_coordinates(
                x1, y1, x2, y2, image_width, image_height, grid_width=144, grid_height=108
            )
            # 转换为四个角点
            corners = bbox_to_corners(grid_x1, grid_y1, grid_x2, grid_y2)
            grid_results[category].append(corners)
    
    # 保存结果到文件
    output_file = os.path.join(save_dir, "detection_results.json")
    with open(output_file, 'w') as f:
        f.write("{\n")
        items = []
        for i, (key, value) in enumerate(grid_results.items()):
            comma = "," if i < len(grid_results) - 1 else ""
            items.append(f'  "{key}":{json.dumps(value, separators=(",", ":"))}{comma}')
        f.write("\n".join(items))
        f.write("\n}")
    
    if verbose:
        print(f"检测结果已保存到: {output_file}")
    return output_file

def detect_and_save(path, save_dir="detection_results", show_results=False, verbose=True):
    """
    检测图像中的目标并输出JSON格式结果
    """

    save_detection_results(detect_objects(path, show_results,verbose=verbose),verbose=verbose)

if __name__ == "__main__":
    # 测试图片路径
    test_image_path = os.path.join(path, './captures/init_mapping.jpg')
    
    # 检测并保存结果
    detect_and_save(test_image_path, show_results=True)
