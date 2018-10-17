#ifndef __HLG_TO_DOLBYPQ_1000NITS_H_INCLUDED__
#define __HLG_TO_DOLBYPQ_1000NITS_H_INCLUDED__


//import "ACESlib.Utilities_Color";



inline float3 HLG_to_DolbyPQ_1000nits( float3 HLG)
{

    float3 PQ = HLG_2_ST2084_1000nits_f3( HLG);

    return PQ;
}

#endif