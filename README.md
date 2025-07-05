# prp-47th-robotic-dog

## camera.py
前往 [海康威视官网](https://www.hikrobotics.com/cn/) 下载并安装 MVS 客户端 (Win/Linux 取决于开发环境)，Linux 下运行`/opt/MVS/bin/MVS.sh`可以在客户端对相机的基本参数（如曝光时间，色彩空间等）进行调整。
Windows 需要复制 `安装目录/MVS/Development/Samples/Python/MvImport`，Linux 需要复制`/opt/MVS/Samples/64/Python/MvImport`到项目文件夹，由于 WSL 不能直接连接 USB 设备，可参考 [解决WSL2开发环境的USB接口问题](https://zhuanlan.zhihu.com/p/636224834) 操作。
直接运行 `camera.py` 测试连接是否正常