__global__ void ReplaceAdjustKernel(int p_Width, int p_Height, float hueRangeA, float hueRangeB, float hueRangeWithRollOffA, float hueRangeWithRollOffB, 
    float hueRotation, float hueRotationGain, float hueMean, float hueRolloff, float satRangeA, float satRangeB, float satAdjust, float satAdjustGain, 
    float satRolloff, float valRangeA, float valRangeB, float valAdjust, float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, 
    float mix, const float* p_Input, float* p_Output)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
   if ((x < p_Width) && (y < p_Height))
   {
    const int index = ((y * p_Width) + x) * 4;
       
	float hcoeff, scoeff, vcoeff;
	float r, g, b, h, s, v;
            
	r = p_Input[index + 0];
	g = p_Input[index + 1];
	b = p_Input[index + 2];
	
    float min = fmin(fmin(r, g), b);
    float max = fmax(fmax(r, g), b);
    v = max;
    float delta = max - min;

    if (max != 0.0f) {
        s = delta / max;
    } else {
        s = 0.0f;
        h = 0.0f;
    }

    if (delta == 0.0f) {
        h = 0.0f;
    } else if (r == max) {
        h = (g - b) / delta;
    } else if (g == max) {
        h = 2 + (b - r) / delta;
    } else {
        h = 4 + (r - g) / delta;
    }
    h *= 1 / 6.0f;
    if (h < 0.0f) {
        h += 1.0f;
    }
	
	float R, G, B;

	h *= 360.0f;
	float h0 = hueRangeA;
	float h1 = hueRangeB;
	float h0mrolloff = hueRangeWithRollOffA;
	float h1prolloff = hueRangeWithRollOffB;
	
	if ( ( h1 < h0 && (h <= h1 || h0 <= h) ) || (h0 <= h && h <= h1) ) {
		hcoeff = 1.0f;
	} else {
		float c0 = 0.0f;
		float c1 = 0.0f;
		
		if ( ( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0) ) {
		c0 = h0 == (h0mrolloff + 360.0f) || h0 == h0mrolloff ? 1.0f : !(( h0 < h0mrolloff && (h <= h0 || h0mrolloff <= h) ) || (h0mrolloff <= h && h <= h0)) ? 0.0f : 
		((h < h0mrolloff ? h + 360.0f : h) - h0mrolloff) / ((h0 < h0mrolloff ? h0 + 360.0f : h0) - h0mrolloff);		
		}
		
		if ( ( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff) ) {
		c1 = !(( h1prolloff < h1 && (h <= h1prolloff || h1 <= h) ) || (h1 <= h && h <= h1prolloff)) ? 0.0f : h1prolloff == h1 ? 1.0f :
		((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - (h < h1 ? h + 360.0f : h)) / ((h1prolloff < h1 ? h1prolloff + 360.0f : h1prolloff) - h1);	
		}
		
		hcoeff = fmax(c0, c1);
	}
	float s0 = satRangeA;
	float s1 = satRangeB;
	float s0mrolloff = s0 - satRolloff;
	float s1prolloff = s1 + satRolloff;
	if ( (s0 <= s) && (s <= s1) ) {
		scoeff = 1.0f;
	} else if ( (s0mrolloff <= s) && (s <= s0) ) {
		scoeff = (s - s0mrolloff) / satRolloff;
	} else if ( (s1 <= s) && (s <= s1prolloff) ) {
		scoeff = (s1prolloff - s) / satRolloff;
	} else {
		scoeff = 0.0f;
	}
	float v0 = valRangeA;
	float v1 = valRangeB;
	float v0mrolloff = v0 - valRolloff;
	float v1prolloff = v1 + valRolloff;
	if ( (v0 <= v) && (v <= v1) ) {
		vcoeff = 1.0f;
	} else if ( (v0mrolloff <= v) && (v <= v0) ) {
		vcoeff = (v - v0mrolloff) / valRolloff;
	} else if ( (v1 <= v) && (v <= v1prolloff) ) {
		vcoeff = (v1prolloff - v) / valRolloff;
	} else {
		vcoeff = 0.0f;
	}
	float coeff = fmin(fmin(hcoeff, scoeff), vcoeff);
	if (coeff <= 0.0f) {
		R = p_Input[index + 0];
		G = p_Input[index + 1];
		B = p_Input[index + 2];
	} else {
	
		float H = (h - hueMean + 180) - (int)(floor((h - hueMean + 180) / 360) * 360) - 180;
		h += coeff * ( hueRotation + (hueRotationGain - 1.0f) * H );
		s += coeff * ( satAdjust + (satAdjustGain - 1.0f) * (s - (s0 + s1) / 2) );
		if (s < 0.0f) {
			s = 0.0f;
		}
		v += coeff * ( valAdjust + (valAdjustGain - 1.0f) * (v - (v0 + v1) / 2) );
		h *= 1 / 360.0f;
	
    if (s == 0.0f) {
        R = G = B = v;
    }
    h *= 6.0f;
    int i = floor(h);
    float f = h - i;
    i = (i >= 0) ? (i % 6) : (i % 6) + 6;
    float p = v * ( 1.0f - s );
    float q = v * ( 1.0f - s * f );
    float t = v * ( 1.0f - s * ( 1.0f - f ));

    if (i == 0){
        R = v;
        G = t;
        B = p;}
    else if (i == 1){
        R = q;
        G = v;
        B = p;}
    else if (i == 2){
        R = p;
        G = v;
        B = t;}
    else if (i == 3){
        R = p;
        G = q;
        B = v;}
	else if (i == 4){
        R = t;
        G = p;
        B = v;}
	else{
		R = v;
        G = p;
        B = q;}
        }
	   	
		float a = OutputAlpha == 0 ? 1.0f : OutputAlpha == 1 ? hcoeff : OutputAlpha == 2 ? scoeff :
		OutputAlpha == 3 ? vcoeff : OutputAlpha == 4 ? fmin(hcoeff, scoeff) : OutputAlpha == 5 ? 
		fmin(hcoeff, vcoeff) : OutputAlpha == 6 ? fmin(scoeff, vcoeff) : fmin(fmin(hcoeff, scoeff), vcoeff);
		p_Output[index + 0] = DisplayAlpha == 1 ? a : R * (1.0f - mix) + p_Input[index + 0] * mix;
		p_Output[index + 1] = DisplayAlpha == 1 ? a : G * (1.0f - mix) + p_Input[index + 1] * mix;
		p_Output[index + 2] = DisplayAlpha == 1 ? a : B * (1.0f - mix) + p_Input[index + 2] * mix;
		p_Output[index + 3] = OutputAlpha != 0 ? a : p_Input[index + 3];
	
   }
}

void RunCudaKernel(int p_Width, int p_Height, float* hueRange, float* hueRangeWithRollOff, 
	float hueRotation, float hueMean, float hueRotationGain, float hueRolloff, float* satRange, 
	float satAdjust, float satAdjustGain, float satRolloff, float* valRange, float valAdjust, 
	float valAdjustGain, float valRolloff, int OutputAlpha, int DisplayAlpha, float mix, 
    const float* p_Input, float* p_Output)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    ReplaceAdjustKernel<<<blocks, threads>>>( p_Width, p_Height, hueRange[0], hueRange[1], hueRangeWithRollOff[0], 
    hueRangeWithRollOff[1], hueRotation, hueRotationGain, hueMean, hueRolloff, satRange[0], satRange[1], satAdjust, 
    satAdjustGain, satRolloff, valRange[0], valRange[1], valAdjust, valAdjustGain, valRolloff, OutputAlpha, DisplayAlpha, 
    mix, p_Input, p_Output);

}
