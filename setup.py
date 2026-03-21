# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_custom_gemm', # 库名字
    ext_modules=[
        CUDAExtension('my_custom_gemm', [
            'binding.cpp', # 翻译官文件
            'matmul.cu',   # 核弹引擎文件
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)