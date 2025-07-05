from camera import HikvisionCamera as Camera
from coordinate_mapper import CoordinateMapper
from predict import detect_and_save
import test
import os
import time
import rospy
from geometry_msgs.msg import PoseStamped
from typing import Tuple

def callback(data):
    rospy.loginfo("Position: x=%.2f, y=%.2f, z=%.2f", 
                  data.pose.position.x, 
                  data.pose.position.y, 
                  data.pose.position.z)

def get_car_position(topic: str, timeout: float = 5.0) -> Tuple[float, float]:
    """
    阻塞等待一个 ROS topic 中的位置信息
    """
    msg = rospy.wait_for_message(topic, PoseStamped, timeout=timeout)
    pos = msg.pose.position
    return (pos.x, pos.y)
    
def main():
    # 创建相机实例
    camera = Camera(device_index=0)

    # 列出所有设备
    devices = camera.list_devices()
    print("可用设备:", devices)

    try:
        # 连接相机
        if camera.connect():
            print("相机连接成功")
            
            camera.capture_rotated_image(file_path="test_img.jpg", angle=-15)

            # 创建坐标映射器实例
            mapper = CoordinateMapper()

            if not mapper.load_mapper("coordinate_mapper.pkl"):
                print("未找到坐标映射器文件，正在初始化...")
                # 检测小车并获取图像坐标
                vehicle_img_coords = mapper.detect_vehicle(path="test_img.jpg", model_path="best.pt", show_results=False)
                print("图像坐标：", vehicle_img_coords)

                # 获取三辆车的 ROS 实际坐标
                print("等待 ROS 小车位置...")
                car_topics = ["/vrpn_client_node/boids_1/pose", "/vrpn_client_node/boids_2/pose", "/vrpn_client_node/boids_3/pose"]
                vehicle_real_coords = []
                
                for topic in car_topics:
                    real_pos = get_car_position(topic)
                    print(f"{topic} -> {real_pos}")
                    vehicle_real_coords.append(real_pos)
                print("实际坐标：", vehicle_real_coords)

                # 初始化坐标映射器
                if len(vehicle_img_coords) == len(vehicle_real_coords):
                    mapper.initialize_transform(vehicle_img_coords, vehicle_real_coords)
                    mapper.save_mapper("coordinate_mapper.pkl")
                    print("坐标映射初始化完成")

                else:
                    print("错误：图像与实际坐标数量不一致")
            else:
                print("已加载坐标映射器")

            while True:
                # 间隔 5s 拍摄单张图片
                filePath = "./captures/image_" + time.strftime("%Y%m%d_%H%M%S") + ".jpg"
                camera.capture_rotated_image(file_path=filePath, angle=-15)
                print(f"已拍摄图片: {filePath}")
                time.sleep(5)

            
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        # 断开连接
        camera.disconnect()
        print("相机已断开连接")

if __name__ == "__main__":
    main()


