#include <torch/extension.h>

// 声明一下我们在 .cu 文件里写的那个发射台函数
void launch_matmul(torch::Tensor a, torch::Tensor b, torch::Tensor c, int width);

// 魔法宏：PYBIND11_MODULE
// "my_custom_gemm" 是你未来在 Python 里 import 的库名字
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 把 C++ 的 launch_matmul 函数，绑定到 Python 里的 "run_v3" 这个名字上
    m.def("run_v3", &launch_matmul, "My V3 Matrix Multiplication (CUDA)");
}

