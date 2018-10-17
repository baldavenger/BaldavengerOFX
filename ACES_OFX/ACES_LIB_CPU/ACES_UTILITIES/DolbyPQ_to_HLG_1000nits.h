#ifndef __DOLBYPQ_TO_HLG_1000NITS_H_INCLUDED__
#define __DOLBYPQ_TO_HLG_1000NITS_H_INCLUDED__


//import "ACESlib.Utilities_Color";



// Conversion of PQ signal to HLG, as detailed in Section 7 of ITU-R BT.2390-0

inline float3 DolbyPQ_to_HLG_1000nits( float3 PQ)
{
    float3 HLG;

    HLG = ST2084_2_HLG_1000nits_f3( PQ);

    return HLG; 
}

#endif