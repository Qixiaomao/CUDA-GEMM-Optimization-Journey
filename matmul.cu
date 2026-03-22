#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

// Define a constant


#define BLOCK_SIZE 16

// 每个工友要啃下的子矩阵大小 Thread Coarsening e.g.2x2
#define THREAD_TILE 2


__global__ void matmul_v3(float *A, float *B, float *C, int width){
    // 全局坐标
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     int Row = blockIdx.y * (BLOCK_SIZE*THREAD_TILE) + ty * THREAD_TILE;
     int Col = blockIdx.x * (BLOCK_SIZE*THREAD_TILE) + tx * THREAD_TILE;
    
    // 开辟 shared memory （黑板
    __shared__ float ds_A[BLOCK_SIZE * THREAD_TILE][BLOCK_SIZE * THREAD_TILE];
    __shared__ float ds_B[BLOCK_SIZE * THREAD_TILE][BLOCK_SIZE * THREAD_TILE];


    // 周天搬运 注意合并访存
    // for 循环
    // 工友拿工牌号记录 
    float c_regs[THREAD_TILE][THREAD_TILE] = {0.0f};
    // 进入大循环
    for(int p = 0; p < width/(BLOCK_SIZE*THREAD_TILE);++p){
        
        
        
        for(int i = 0; i < THREAD_TILE;++i){ // y 方向上的跳跃
            for(int j = 0; j < THREAD_TILE;++j){  // x 方向上的跳跃
               // 算出在32X32黑板上的落点坐标
               //ty 0~15, i*BLOCK_SIZE 是跳跃的步长 （0 or 16)
               int shared_y = ty + i * BLOCK_SIZE;
               int shared_x = tx + j * BLOCK_SIZE;

               // 算出对应的Global Memory 绝对坐标
               int global_A_row = blockIdx.y * (BLOCK_SIZE * THREAD_TILE) + shared_y;
               int global_A_col = p*(BLOCK_SIZE * THREAD_TILE) + shared_x;
               
               int global_B_row = p*(BLOCK_SIZE * THREAD_TILE) + shared_y;
               int global_B_col = blockIdx.x * (BLOCK_SIZE * THREAD_TILE) + shared_x;

               // 开始搬砖
                ds_A[shared_y][shared_x] = A[global_A_row*width + global_A_col];
                ds_B[shared_y][shared_x] = B[global_B_row*width + global_B_col];
            }
        }

           __syncthreads(); // 同步等1024块砖放好


           // 内景杀伐 shared_memory 极速计算
           // 顺着黑板长度（32）扫一遍

          float a_regs[THREAD_TILE];
          float b_regs[THREAD_TILE];
           for(int k = 0; k < BLOCK_SIZE*THREAD_TILE;++k){
           
            // 抓取A矩阵的2个数据 负责2行
            for(int y = 0;y < THREAD_TILE; ++y){
                a_regs[y] = ds_A[ty*THREAD_TILE+y][k];
            }

            // B矩阵的2个数据 负责2列
            for(int x = 0; x < THREAD_TILE;++x){
               b_regs[x] = ds_B[k][tx*THREAD_TILE+x];
            }

            // 交叉相乘，累加到c_regs
            for(int y=0; y < THREAD_TILE; ++y){
                for(int x=0; x < THREAD_TILE;++x){
                    c_regs[y][x] += a_regs[y] * b_regs[x];
                }
            }
           } // 结束k循环（黑板扫视完毕

           __syncthreads(); // 同步准备下一次大循环
    }

    // 收剑入鞘 写回结果
    // 遍历自己裤兜里的2x2 （THREAD_TILE X THREAD_TILE）个结果
   for(int y=0; y < THREAD_TILE; ++y){
    for(int x=0; x < THREAD_TILE;++x){
        // 算出砖在global memory的绝对物理坐标
        // Row and Col 是左上角基准点，加上局部偏移量 y and x
        int global_row = Row + y;
        int global_col = Col + x;

        // 安全检查：只写矩阵内部，边缘悬空丢弃，绝不越界！
        if(global_row < width && global_col < width){
            // 从二维裤兜 regs[y][x] 掏出结果，写回矩阵的一维地址
            C[global_row * width + global_col] = c_regs[y][x];
        }
    }
   }

}



// 这个函数就是供外部调用的发射台
void launch_matmul(torch::Tensor a, torch::Tensor b, torch::Tensor c, int width) {
    // 派兵布阵
    dim3 dimBlock(16, 16);
    // 注意：V3.0 里一个 Block 覆盖 32x32 的面积，所以 Grid 数量要除以 32
    dim3 dimGrid(width / 32, width / 32); 

    // 提取 PyTorch Tensor 底层的极其原始的 C++ 数据指针！
    float* d_a = a.data_ptr<float>();
    float* d_b = b.data_ptr<float>();
    float* d_c = c.data_ptr<float>();

    // 点火！启动已写好的 V3 Kernel
    matmul_v3<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);
}

