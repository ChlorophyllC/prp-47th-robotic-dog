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

def bbox_to_corners(x1, y1, x2, y2):
    """
    将边界框转换为四个角点坐标
    """
    return [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]

def detect_and_save(path, save_dir="detection_results", show_results=False):
    """
    检测图像中的目标并输出JSON格式结果
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # YOLO检测
    frame = cv2.imread(path)
    if frame is None:
        print(f"无法读取图像: {path}")
        return
    
    results = model.predict(frame, imgsz=640, conf=0.5)
    
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
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            
            # 转换为网格坐标
            grid_x1, grid_y1, grid_x2, grid_y2 = convert_to_grid_coordinates(
                x1, y1, x2, y2, image_width, image_height, grid_width=72, grid_height=54
            )
            
            # 转换为四个角点
            corners = bbox_to_corners(grid_x1, grid_y1, grid_x2, grid_y2)
            
            # 根据类别分类
            class_name = class_names[cls_id]
            if class_name == 'Red':
                detection_results["obstacle"].append(corners)
            elif class_name == 'Green':
                detection_results["destination"].append(corners)
            elif class_name == 'Vehicle':
                detection_results["all_vehicles"].append(corners)
    
    if show_results:
        # 绘制检测框并显示
        annotated_frame = draw_boxes(frame.copy(), results, class_names)
        
        # 显示实时结果
        cv2.imshow("YOLOv11 Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 保存结果到文件
    output_file = os.path.join(save_dir, "detection_results.json")
    with open(output_file, 'w') as f:
        f.write("{\n")
        items = []
        for i, (key, value) in enumerate(detection_results.items()):
            comma = "," if i < len(detection_results) - 1 else ""
            items.append(f'  "{key}":{json.dumps(value, separators=(",", ":"))}{comma}')
        f.write("\n".join(items))
        f.write("\n}")
    
    print(f"检测结果已保存到: {output_file}")

if __name__ == "__main__":
    # 测试图片路径
    test_image_path = os.path.join(path, 'test_image.jpg')
    
    # 检测并保存结果
    detect_and_save(test_image_path, show_results=True)