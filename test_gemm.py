import torch
import my_custom_gemm
import time

width = 1024
print(f"初始化 {width}x{width} 的矩阵...")
a = torch.randn(width, width, device='cuda', dtype=torch.float32)
b = torch.randn(width, width, device='cuda', dtype=torch.float32)
c_custom = torch.zeros(width, width, device='cuda', dtype=torch.float32)

print("=== GPU 工友热身运动 (Warm-up) ===")
# 先随便跑几次，让 GPU 把 Context 初始化好，把缓存打热
for _ in range(5):
    _ = torch.matmul(a, b)
    my_custom_gemm.run_v3(a, b, c_custom, width)
torch.cuda.synchronize() # 等大家都热身完

print("=== 真正的巅峰对决 (取 10 次平均) ===")

# 测 PyTorch 官方
start_time = time.time()
for _ in range(10):
    c_torch = torch.matmul(a, b)
torch.cuda.synchronize()
torch_time = (time.time() - start_time) * 1000 / 10

# 测你的 V3.0
start_time = time.time()
for _ in range(10):
    my_custom_gemm.run_v3(a, b, c_custom, width)
torch.cuda.synchronize()
custom_time = (time.time() - start_time) * 1000 / 10

# 验证精度
is_correct = torch.allclose(c_torch, c_custom, atol=1e-3)

print("="*40)
print(f"验证结果: {'✅ 完美通过 (Success!)' if is_correct else '❌ 精度对不上 (Failed!)'}")
print(f"PyTorch 官方 (cuBLAS) 平均耗时: {torch_time:.3f} ms")
print(f"你的 V3.0 平均耗时: {custom_time:.3f} ms")
print("="*40)