import numpy as np
import cv2
from ultralytics import YOLO
from typing import Tuple, List, Optional
import pickle
import os

class_names = ['Yellow', 'Red', 'Green', 'Blue', 'Vehicle', 'Ball']

class CoordinateMapper:
    """
    使用仿射变换将图片坐标映射到现实坐标的类
    """
    
    def __init__(self):
        self.transform_matrix = None
        self.inverse_transform_matrix = None
        self.is_initialized = False
    
    def initialize_transform(self, 
                           image_coords: List[Tuple[float, float]], 
                           real_coords: List[Tuple[float, float]]) -> bool:
        """
        初始化坐标变换矩阵
        
        Args:
            image_coords: 图片坐标列表 [(x1, y1), (x2, y2), (x3, y3)]
            real_coords: 对应的现实坐标列表 [(X1, Y1), (X2, Y2), (X3, Y3)]
            
        Returns:
            bool: 是否成功初始化
        """
        if len(image_coords) != 3 or len(real_coords) != 3:
            raise ValueError("需要提供3个对应的坐标点")
        
        try:
            # 构建仿射变换矩阵
            # 仿射变换: [X, Y, 1] = [x, y, 1] * T
            # 其中 T 是 3x3 的变换矩阵
            
            # 图片坐标矩阵 (齐次坐标)
            img_matrix = np.array([
                [image_coords[0][0], image_coords[0][1], 1],
                [image_coords[1][0], image_coords[1][1], 1],
                [image_coords[2][0], image_coords[2][1], 1]
            ])
            
            # 现实坐标矩阵 (齐次坐标)
            real_matrix = np.array([
                [real_coords[0][0], real_coords[0][1], 1],
                [real_coords[1][0], real_coords[1][1], 1],
                [real_coords[2][0], real_coords[2][1], 1]
            ])
            
            # 计算变换矩阵: T = inv(img_matrix) * real_matrix
            img_matrix_inv = np.linalg.inv(img_matrix)
            self.transform_matrix = img_matrix_inv @ real_matrix
            
            self.is_initialized = True
            return True
            
        except np.linalg.LinAlgError:
            print("错误：提供的三个点共线，无法建立有效的仿射变换")
            return False
        except Exception as e:
            print(f"初始化失败：{e}")
            return False
    
    def map_to_real_coords(self, image_coord: Tuple[float, float]) -> Tuple[float, float]:
        """
        将图片坐标映射到现实坐标（输入为一个(x, y)元组）

        Args:
            image_coord: 图片坐标 (x, y)

        Returns:
            Tuple[float, float]: 现实坐标 (X, Y)
        """
        if not self.is_initialized:
            raise RuntimeError("坐标映射器尚未初始化，请先调用 initialize_transform")

        x, y = image_coord
        img_point = np.array([x, y, 1])
        real_point = img_point @ self.transform_matrix
        return (real_point[0], real_point[1])
    
    def batch_map_to_real_coords(self, image_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        批量将图片坐标映射到现实坐标
        
        Args:
            image_coords: 图片坐标列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            List[Tuple[float, float]]: 现实坐标列表 [(X1, Y1), (X2, Y2), ...]
        """
        if not self.is_initialized:
            raise RuntimeError("坐标映射器尚未初始化，请先调用 initialize_transform")
        
        # 批量处理以减少重复计算
        img_points = np.array([[x, y, 1] for x, y in image_coords])
        real_points = img_points @ self.transform_matrix
        
        return [(point[0], point[1]) for point in real_points]
    
    def batch_map_to_image_coords(self, real_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        批量将现实坐标映射回图片坐标
        
        Args:
            real_coords: 现实坐标列表 [(X1, Y1), (X2, Y2), ...]
            
        Returns:
            List[Tuple[float, float]]: 图片坐标列表 [(x1, y1), (x2, y2), ...]
        """
        if not self.is_initialized:
            raise RuntimeError("坐标映射器尚未初始化，请先调用 initialize_transform")
        
        # 计算逆变换矩阵（如果尚未计算）
        if not hasattr(self, 'inverse_transform_matrix'):
            self.inverse_transform_matrix = np.linalg.inv(self.transform_matrix)
        
        # 批量处理以提高效率
        real_points = np.array([[x, y, 1] for x, y in real_coords])
        img_points = real_points @ self.inverse_transform_matrix

        return [(point[0], point[1]) for point in img_points]

    def get_transform_info(self) -> Optional[np.ndarray]:
        """
        获取变换矩阵信息
        
        Returns:
            Optional[np.ndarray]: 变换矩阵，如果未初始化则返回None
        """
        return self.transform_matrix if self.is_initialized else None

    def detect_vehicle(self, path, model_path="best.pt", show_results=False) -> List[Tuple[float, float]]:
        """检测图像中的小车目标"""
        frame = cv2.imread(path)
        if frame is None:
            print(f"无法读取图像: {path}")
            return None

        image_height, image_width = frame.shape[:2]

        model = YOLO(model_path)
        results = model.predict(frame, imgsz=640, conf=0.5)

        vehicle_coords = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # 获取框坐标和类别
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # 根据类别分类
                class_name = class_names[cls_id]
                if class_name != "Vehicle":
                    continue  # 不是vehicle类跳过

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if center_x > image_width or center_y > image_height:
                    print(f"⚠ 警告：检测坐标超出范围 ({center_x:.1f}, {center_y:.1f}) > ({image_width}, {image_height})")

                vehicle_coords.append((center_x, center_y))

                if show_results:
                    print(f"检测到小车: 中心坐标 ({center_x:.1f}, {center_y:.1f})")

        if len(vehicle_coords) < 3:
            print(f"⚠ 只检测到 {len(vehicle_coords)} 个小车，无法建立坐标映射")
            return None

        vehicle_coords = sorted(vehicle_coords, key=lambda x: x[1])[:3]
        return vehicle_coords

    def save_mapper(self, filepath: str) -> bool:
        """
        保存坐标映射器到文件
        
        Args:
            filepath: 保存文件路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.is_initialized:
            print("错误：坐标映射器尚未初始化，无法保存")
            return False
        
        try:
            mapper_data = {
                'transform_matrix': self.transform_matrix,
                'is_initialized': self.is_initialized
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(mapper_data, f)
            
            print(f"坐标映射器已保存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    @classmethod
    def load_mapper(cls, filepath: str) -> Optional['CoordinateMapper']:
        """
        从文件加载坐标映射器
        
        Args:
            filepath: 文件路径
            
        Returns:
            Optional[CoordinateMapper]: 加载的坐标映射器，失败时返回None
        """
        if not os.path.exists(filepath):
            print(f"错误：文件不存在 {filepath}")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                mapper_data = pickle.load(f)
            
            mapper = cls()
            mapper.transform_matrix = mapper_data['transform_matrix']
            mapper.is_initialized = True
            
            if mapper.is_initialized:
                mapper.inverse_transform_matrix = np.linalg.inv(mapper.transform_matrix)
                
            print(f"坐标映射器已从文件加载: {filepath}")
            return mapper
            
        except Exception as e:
            print(f"加载失败: {e}")
            return None


# 使用示例
if __name__ == "__main__":
    # 示例：三个vehicle的坐标
    vehicle_img_coords = [
        (100, 200),  # vehicle 1 在图片上的坐标
        (300, 150),  # vehicle 2 在图片上的坐标
        (200, 400)   # vehicle 3 在图片上的坐标
    ]
    
    vehicle_real_coords = [
        (10.5, 20.3),  # vehicle 1 的现实坐标
        (25.2, 15.8),  # vehicle 2 的现实坐标
        (18.9, 35.1)   # vehicle 3 的现实坐标
    ]
    
    # 创建坐标映射器
    mapper = CoordinateMapper()
    mapper = mapper.load_mapper('coordinate_mapper.pkl')
    print(mapper.transform_matrix)
    # mapper.initialize_transform(vehicle_img_coords, vehicle_real_coords)
    # mapper.detect_vehicle(path='./captures/init_mapping.jpg',show_results=True)
    # 测试映射
    test_img_point = (250, 250)
    real_point = mapper.map_to_real_coords(test_img_point)
    print(f"图片坐标 {test_img_point} 映射到现实坐标: {real_point}")
    
    # 批量映射测试
    test_points = [(150, 180), (280, 320), (180, 280)]
    real_points = mapper.batch_map_to_real_coords(test_points)
    print(f"批量映射结果:")
    for img_p, real_p in zip(test_points, real_points):
        print(f"  {img_p} -> {real_p}")
    
    # 验证已知点的映射精度
    print(f"\n验证映射精度:")
    for i, (img_p, expected_real_p) in enumerate(zip(vehicle_img_coords, vehicle_real_coords)):
        mapped_real_p = mapper.map_to_real_coords(img_p)
        error = ((mapped_real_p[0] - expected_real_p[0])**2 + (mapped_real_p[1] - expected_real_p[1])**2)**0.5
        print(f"  Vehicle {i+1}: 误差 = {error:.6f}")