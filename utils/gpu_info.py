import torch


def gpu_info():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print('CUDA is available!')
    else:
        print('CUDA is not available.')

    # 获取可用的 CUDA 设备数量
    device_count = torch.cuda.device_count()
    print(f'Number of available CUDA devices: {device_count}')

    # 获取默认 CUDA 设备的索引
    default_device_index = torch.cuda.current_device()
    print(f'Default CUDA device index: {default_device_index}')

    # 获取指定索引的 CUDA 设备名称
    device_name = torch.cuda.get_device_name(default_device_index)
    print(f'CUDA device name: {device_name}')

    # 获取指定索引的 CUDA 设备属性
    device_prop = torch.cuda.get_device_properties(default_device_index)
    print(f'CUDA device properties: {device_prop}')