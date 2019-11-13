#include "Types.hpp"
namespace SIPL {

// float2
float float2::distance(float2 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y));
}
float float2::dot(float2 other) const {
    return x*other.x+y*other.y;
}
float float2::distance(int2 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y));
}
float float2::dot(int2 other) const {
    return x*other.x+y*other.y;
}
bool float2::operator==(int2 other) const {
	return x==other.x && y==other.y;
}
bool float2::operator==(float2 other) const {
	return x==other.x && y==other.y;
}

// float3
float float3::distance(float3 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y)+(z-other.z)*(z-other.z));
}
float float3::dot(float3 other) const {
    return x*other.x+y*other.y+z*other.z;
}
float float3::distance(int3 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y)+(z-other.z)*(z-other.z));
}
float float3::dot(int3 other) const {
    return x*other.x+y*other.y+z*other.z;
}
bool float3::operator==(int3 other) const {
	return x==other.x && y==other.y && z==other.z;
}
bool float3::operator==(float3 other) const {
	return x==other.x && y==other.y && z==other.z;
}

// int2
float int2::distance(float2 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y));
}
float int2::dot(float2 other) const {
    return x*other.x+y*other.y;
}
float int2::distance(int2 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y));
}
float int2::dot(int2 other) const {
    return x*other.x+y*other.y;
}
bool int2::operator==(int2 other) const {
	return x==other.x && y==other.y;
}
bool int2::operator==(float2 other) const {
	return x==other.x && y==other.y;
}


// int3
float int3::distance(float3 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y)+(z-other.z)*(z-other.z));
}
float int3::dot(float3 other) const {
    return x*other.x+y*other.y+z*other.z;
}
float int3::distance(int3 other) const {
    return sqrt((x-other.x)*(x-other.x)+(y-other.y)*(y-other.y)+(z-other.z)*(z-other.z));
}
float int3::dot(int3 other) const {
    return x*other.x+y*other.y+z*other.z;
}
bool int3::operator==(int3 other) const {
	return x==other.x && y==other.y && z==other.z;
}
bool int3::operator==(float3 other) const {
	return x==other.x && y==other.y && z==other.z;
}


};
