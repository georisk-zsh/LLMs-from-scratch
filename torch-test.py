import torch

print("=== PyTorch MPS 测试 ===")

# 1. 测试 PyTorch 基本安装
print(f"PyTorch 版本: {torch.__version__}")

# 2. 测试 MPS 可用性
print(f"MPS 后端可用: {torch.backends.mps.is_available()}")
print(f"MPS 后端已构建: {torch.backends.mps.is_built()}")

# 3. 测试设备
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(f"MPS 设备: {mps_device}")
    
    # 4. 测试张量创建和计算
    x = torch.randn(3, 3, device=mps_device)
    y = torch.randn(3, 3, device=mps_device)
    z = x + y  # 简单的 GPU 计算
    
    print(f"张量创建成功: {x.device}")
    print(f"计算完成: {z.device}")
    print(f"张量形状: {z.shape}")
    
    # 5. 测试与 CPU 的差异
    x_cpu = x.cpu()
    diff = torch.abs(x_cpu - x.cpu())  # 都转到 CPU 比较
    print(f"CPU 和 MPS 结果差异: {diff.max().item()}")
    
else:
    print("❌ MPS 不可用，请检查安装")