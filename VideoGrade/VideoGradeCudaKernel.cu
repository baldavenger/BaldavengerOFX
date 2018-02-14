__global__ void VideoGradeAdjustKernel(int p_Width, int p_Height, float p_SwitchA, float p_GainL, float p_GainLA, float p_GainG, float p_GainGa, float p_GainGb, float p_GainGG, float p_GainGA, float p_GainO, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;

   if ((x < p_Width) && (y < p_Height))
   {
       const int index = ((y * p_Width) + x) * 4;
		
	  float GR = p_Input[index + 0] >= p_GainGA ? (p_Input[index + 0] - p_GainGA) * p_GainGG  + p_GainGA: p_Input[index + 0];
	  float LR = GR <= p_GainLA ? (((GR / p_GainLA) + (p_GainL * (1 - (GR / p_GainLA)))) * p_GainLA) + p_GainO: GR + p_GainO;
	  float Prl = LR >= p_GainGa && LR <= p_GainGb ? powf((LR - p_GainGa) / (p_GainGb - p_GainGa), 1.0/p_GainG) * (p_GainGb - p_GainGa) + p_GainGa : LR;
	  float Pru = LR >= p_GainGa && LR <= p_GainGb ? (1.0 - powf(1.0 - (LR - p_GainGa) / (p_GainGb - p_GainGa), p_GainG)) * (p_GainGb - p_GainGa) + p_GainGa : LR;
	  float R = p_SwitchA == 1.0f ? Pru : Prl;
	  
	  float GG = p_Input[index + 1] >= p_GainGA ? (p_Input[index + 1] - p_GainGA) * p_GainGG  + p_GainGA: p_Input[index + 1];
	  float LG = GG <= p_GainLA ? (((GG / p_GainLA) + (p_GainL * (1 - (GG / p_GainLA)))) * p_GainLA) + p_GainO: GG + p_GainO;
	  float Pgl = LG >= p_GainGa && LG <= p_GainGb ? powf((LG - p_GainGa) / (p_GainGb - p_GainGa), 1.0/p_GainG) * (p_GainGb - p_GainGa) + p_GainGa : LG;
	  float Pgu = LG >= p_GainGa && LG <= p_GainGb ? (1.0 - powf(1.0 - (LG - p_GainGa) / (p_GainGb - p_GainGa), p_GainG)) * (p_GainGb - p_GainGa) + p_GainGa : LG;
	  float G = p_SwitchA == 1.0f ? Pgu : Pgl;
	  
	  float GB = p_Input[index + 2] >= p_GainGA ? (p_Input[index + 2] - p_GainGA) * p_GainGG  + p_GainGA: p_Input[index + 2];
	  float LB = GB <= p_GainLA ? (((GB / p_GainLA) + (p_GainL * (1 - (GB / p_GainLA)))) * p_GainLA) + p_GainO: GB + p_GainO;
	  float Pbl = LB >= p_GainGa && LB <= p_GainGb ? powf((LB - p_GainGa) / (p_GainGb - p_GainGa), 1.0/p_GainG) * (p_GainGb - p_GainGa) + p_GainGa : LB;
	  float Pbu = LB >= p_GainGa && LB <= p_GainGb ? (1.0 - powf(1.0 - (LB - p_GainGa) / (p_GainGb - p_GainGa), p_GainG)) * (p_GainGb - p_GainGa) + p_GainGa : LB;
	  float B = p_SwitchA == 1.0f ? Pbu : Pbl;
		
							
       p_Output[index + 0] = R;
       p_Output[index + 1] = G;
       p_Output[index + 2] = B;
       p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    VideoGradeAdjustKernel<<<blocks, threads>>>(p_Width, p_Height, p_Switch[0], p_Gain[0], p_Gain[1], p_Gain[2], 
    p_Gain[3], p_Gain[4], p_Gain[5], p_Gain[6], p_Gain[7], p_Input, p_Output);
}
