__global__ void SimpleAdjustKernel(int p_Width, int p_Height, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
      const int index = ((y * p_Width) + x) * 4;
       																												
	 p_Output[index + 0] = p_Input[index + 0];
	 p_Output[index + 1] = p_Input[index + 1];
	 p_Output[index + 2] = p_Input[index + 2];
	 p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    SimpleAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Input, p_Output);
}
