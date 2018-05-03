#ifndef _HUECONVERGECUDABLUR_H_INCLUDED_
#define _HUECONVERGECUDABLUR_H_INCLUDED_

#define BLOCK_DIM 32

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void d_transpose(float *idata, float *odata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = (yIndex * width + xIndex);
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = (yIndex * height + xIndex) * 4;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void d_recursiveGaussian(float *id, float *od, int w, int h, float blur)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
    const float nsigma = blur < 0.1f ? 0.1f : blur,
	alpha = 1.695f / nsigma,
	ema = (float)exp(-alpha);
	float ema2 = (float)exp(-2*alpha),
	b1 = -2*ema,
	b2 = ema2;
	float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;
	const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
	a0 = k;
	a1 = k*(alpha-1)*ema;
	a2 = k*(alpha+1)*ema;
	a3 = -k*ema2;
	coefp = (a0+a1)/(1+b1+b2);
	coefn = (a2+a3)/(1+b1+b2);

    if (x >= w) return;

    id += x * 4;    // advance pointers to correct column
    od += x;

    // forward pass
    float xp, yp, yb;
    xp = *id;
    yb = coefp*xp;
    yp = yb;

    for (int y = 0; y < h; y++)
    {
        float xc = *id;
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
        *od = yc;
        id += w * 4;
        od += w;    // move to next row
        xp = xc;
        yb = yp;
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w * 4;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    float xn, xa, yn, ya;
    
    xn = xa = *id;
    yn = coefn*xn;
    ya = yn;

    for (int y = h-1; y >= 0; y--)
    {
        float xc = *id;
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        *od = *od + yc;
        id -= w * 4;
        od -= w;  // move to previous row
    }
}

#endif // #ifndef _HUECONVERGECUDABLUR_H_INCLUDED_