import cv2
import sys
import copy
import numpy as np
import time
import threading
from datetime import datetime
from ctypes import *

sys.path.append("./MvImport")
from MvCameraControl_class import *


class HikvisionCamera:
    """海康威视相机控制类"""
    
    def __init__(self, device_index=0):
        """
        初始化相机
        
        Args:
            device_index (int): 设备索引，默认为0（第一个设备）
        """
        self.device_index = device_index
        self.cam = None
        self.device_list = None
        self.payload_size = 0
        self.is_connected = False
        self.is_grabbing = False
        self.capture_thread = None
        self.stop_capture = False
        
    def list_devices(self):
        """列出所有可用设备"""
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        
        # 枚举设备
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            print(f"枚举设备失败! ret[0x{ret:x}]")
            return []
        
        if device_list.nDeviceNum == 0:
            print("未找到设备!")
            return []
        
        print(f"找到 {device_list.nDeviceNum} 个设备!")
        devices = []
        
        for i in range(0, device_list.nDeviceNum):
            mvcc_dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            device_info = {"index": i}
            
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print(f"\nGigE设备: [{i}]")
                # 获取设备名称
                model_name = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    model_name = model_name + chr(per)
                print(f"设备型号: {model_name}")
                
                # 获取IP地址
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
                print(f"当前IP: {ip}")
                
                device_info.update({
                    "type": "GigE",
                    "model_name": model_name.strip(),
                    "ip": ip
                })
                
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print(f"\nUSB3.0设备: [{i}]")
                # 获取设备名称
                model_name = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    model_name = model_name + chr(per)
                print(f"设备型号: {model_name}")
                
                # 获取序列号
                serial_number = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    serial_number = serial_number + chr(per)
                print(f"序列号: {serial_number}")
                
                device_info.update({
                    "type": "USB3.0",
                    "model_name": model_name.strip(),
                    "serial_number": serial_number.strip()
                })
            
            devices.append(device_info)
        
        return devices
    
    def connect(self):
        """连接相机"""
        if self.is_connected:
            print("相机已经连接!")
            return True
        
        # 获取设备列表
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            print(f"枚举设备失败! ret[0x{ret:x}]")
            return False
        
        if device_list.nDeviceNum == 0:
            print("未找到设备!")
            return False
        
        if self.device_index >= device_list.nDeviceNum:
            print("设备索引超出范围!")
            return False
        
        # 创建相机实例
        self.cam = MvCamera()
        
        # 选择设备并创建句柄
        st_device_list = cast(device_list.pDeviceInfo[self.device_index], POINTER(MV_CC_DEVICE_INFO)).contents
        self.device_list = st_device_list
        
        ret = self.cam.MV_CC_CreateHandle(st_device_list)
        if ret != 0:
            print(f"创建句柄失败! ret[0x{ret:x}]")
            return False
        
        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"打开设备失败! ret[0x{ret:x}]")
            return False
        
        # 探测网络最佳包大小(只对GigE相机有效)
        if st_device_list.nTLayerType == MV_GIGE_DEVICE:
            packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    print(f"警告: 设置包大小失败! ret[0x{ret:x}]")
            else:
                print(f"警告: 获取包大小失败! ret[0x{packet_size:x}]")
        
        # 设置触发模式为off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"设置触发模式失败! ret[0x{ret:x}]")
            return False
        
        # 获取数据包大小
        st_param = MVCC_INTVALUE()
        memset(byref(st_param), 0, sizeof(MVCC_INTVALUE))
        
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", st_param)
        if ret != 0:
            print(f"获取载荷大小失败! ret[0x{ret:x}]")
            return False
        
        self.payload_size = st_param.nCurValue
        self.is_connected = True
        print("相机连接成功!")
        return True
    
    def start_grabbing(self):
        """开始取流"""
        if not self.is_connected:
            print("相机未连接!")
            return False
        
        if self.is_grabbing:
            print("相机已经在取流中!")
            return True
        
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"开始取流失败! ret[0x{ret:x}]")
            return False
        
        self.is_grabbing = True
        print("开始取流成功!")
        return True
    
    def stop_grabbing(self):
        """停止取流"""
        if not self.is_grabbing:
            return True
        
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print(f"停止取流失败! ret[0x{ret:x}]")
            return False
        
        self.is_grabbing = False
        print("停止取流成功!")
        return True
    
    def rotate_image(self, image, angle):
        """
        旋转图像
        
        Args:
            image (numpy.ndarray): 输入图像
            angle (float): 旋转角度（逆时针）
        
        Returns:
            numpy.ndarray: 旋转后的图像
        """
        if angle == 0:
            return image
        
        # 获取图像中心
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 执行旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated_image
    
    def capture_image(self, file_path=None, timeout=1000):
        """
        拍摄一张图片
        
        Args:
            file_path (str): 保存路径，如果为None则自动生成
            timeout (int): 超时时间（毫秒）
        
        Returns:
            bool: 是否成功
        """
        if not self.is_connected:
            print("相机未连接!")
            return False
        
        if not self.is_grabbing:
            if not self.start_grabbing():
                return False
        
        # 生成默认文件名
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"image_{timestamp}.jpg"
        
        st_frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(st_frame_info), 0, sizeof(st_frame_info))
        data_buf = (c_ubyte * self.payload_size)()
        
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(data_buf), self.payload_size, st_frame_info, timeout)
        if ret == 0:
            print(f"获取一帧: 宽度[{st_frame_info.nWidth}], 高度[{st_frame_info.nHeight}], 帧号[{st_frame_info.nFrameNum}]")
            
            # 保存为JPEG
            rgb_size = st_frame_info.nWidth * st_frame_info.nHeight * 3
            st_convert_param = MV_SAVE_IMAGE_PARAM_EX()
            st_convert_param.nWidth = st_frame_info.nWidth
            st_convert_param.nHeight = st_frame_info.nHeight
            st_convert_param.pData = data_buf
            st_convert_param.nDataLen = st_frame_info.nFrameLen
            st_convert_param.enPixelType = st_frame_info.enPixelType
            st_convert_param.nImageLen = st_convert_param.nDataLen
            st_convert_param.nJpgQuality = 70
            st_convert_param.enImageType = MV_Image_Jpeg
            st_convert_param.pImageBuffer = (c_ubyte * rgb_size)()
            st_convert_param.nBufferSize = rgb_size
            
            ret = self.cam.MV_CC_SaveImageEx2(st_convert_param)
            if ret != 0:
                print(f"转换像素失败! ret[0x{ret:x}]")
                return False
            
            # 保存文件
            try:
                with open(file_path, 'wb') as file_open:
                    buffer_ptr = cast(st_convert_param.pImageBuffer, POINTER(c_ubyte * st_convert_param.nImageLen))
                    file_open.write(buffer_ptr.contents)

                print(f"保存图片成功: {file_path}")
                return True
            except Exception as e:
                print(f"保存文件失败: {e}")
                return False
        else:
            print(f"获取图像失败! ret[0x{ret:x}]")
            return False
    
    def capture_rotated_image(self, file_path=None, angle=0, timeout=1000):
        """
        拍摄一张旋转后的图片
        
        Args:
            file_path (str): 保存路径，如果为None则自动生成
            angle (float): 旋转角度（逆时针）
            timeout (int): 超时时间（毫秒）
        
        Returns:
            bool: 是否成功
        """
        if not self.is_connected:
            print("相机未连接!")
            return False
        
        if not self.is_grabbing:
            if not self.start_grabbing():
                return False
        
        # 生成默认文件名
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"rotated_image_{timestamp}.jpg"
        
        # 获取图像
        if not self.capture_image(file_path, timeout):
            return False
        
        # 读取图像并旋转
        image = cv2.imread(file_path)
        rotated_image = self.rotate_image(image, angle)
        
        # 保存旋转后的图像
        cv2.imwrite(file_path, rotated_image)
        print(f"保存旋转后的图片成功: {file_path}")
        
        return True

    def start_interval_capture(self, interval_seconds, angle=0, save_dir="./", name_prefix="image"):
        """
        开始间隔拍摄
        
        Args:
            interval_seconds (float): 拍摄间隔时间（秒）
            angle (float): 旋转角度（逆时针）
            save_dir (str): 保存目录
            name_prefix (str): 文件名前缀
        """
        if self.capture_thread and self.capture_thread.is_alive():
            print("间隔拍摄已经在运行中!")
            return
        
        self.stop_capture = False
        self.capture_thread = threading.Thread(
            target=self._interval_capture_worker,
            args=(interval_seconds, angle, save_dir, name_prefix)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print(f"开始间隔拍摄，间隔时间: {interval_seconds}秒")
    
    def stop_interval_capture(self):
        """停止间隔拍摄"""
        if self.capture_thread and self.capture_thread.is_alive():
            self.stop_capture = True
            self.capture_thread.join()
            print("停止间隔拍摄")
    
    def _interval_capture_worker(self, interval_seconds, angle, save_dir, name_prefix):
        """间隔拍摄工作线程"""
        import os
        
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        while not self.stop_capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_dir, f"{name_prefix}_{timestamp}.jpg")
            
            if self.capture_rotated_image(file_path, angle=angle):
                print(f"间隔拍摄成功: {file_path}")
            else:
                print("间隔拍摄失败")
            
            # 等待指定时间
            time.sleep(interval_seconds)
    
    def disconnect(self):
        """断开相机连接"""
        if not self.is_connected:
            return
        
        # 停止间隔拍摄
        self.stop_interval_capture()
        
        # 停止取流
        self.stop_grabbing()
        
        # 关闭设备
        if self.cam:
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                print(f"关闭设备失败! ret[0x{ret:x}]")
            
            # 销毁句柄
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                print(f"销毁句柄失败! ret[0x{ret:x}]")
        
        self.is_connected = False
        self.cam = None
        print("相机断开连接!")
    
    def __del__(self):
        """析构函数"""
        self.disconnect()


# 使用示例
if __name__ == "__main__":
    # 创建相机实例
    camera = HikvisionCamera(device_index=0)
    
    # 列出所有设备
    devices = camera.list_devices()
    
    try:
        # 连接相机
        if camera.connect():
            # 拍摄单张图片
            camera.capture_image("test_image.jpg")
            
            # 开始间隔拍摄（每5秒拍一张）
            camera.start_interval_capture(interval_seconds=5.0, angle = -15, save_dir="./captures", name_prefix="auto")
            
            # 运行30秒后停止
            time.sleep(30)
            camera.stop_interval_capture()
            
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        # 断开连接
        camera.disconnect()