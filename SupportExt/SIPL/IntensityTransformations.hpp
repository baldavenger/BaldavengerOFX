#ifndef INTENSITYTRANSFORMATIONS_HPP_
#define INTENSITYTRANSFORMATIONS_HPP_

#include "Types.hpp"
#include "Exceptions.hpp"
#include <typeinfo>
#include <iostream>
namespace SIPL {

enum TransformationType { DEFAULT, HOUNSEFIELD, AVERAGE, NORMALIZED, CUSTOM };

template <class T>
inline float toSingleValue(T value) {
    return (float)(value);
};

template <>
inline float toSingleValue<color_uchar>(color_uchar value) {
    return (float)(0.33f*(value.red+value.blue+value.green));
};

template <>
inline float toSingleValue<float3>(float3 value) {
    return (float)(0.33f*(value.x+value.y+value.z));
};

template <>
inline float toSingleValue<float2>(float2 value) {
    return (float)(0.33f*(value.x+value.y));
};

template <class T>
inline float3 toVectorData(T value) {
    float3 v(value,value,value);
    return v;
}

template <>
inline float3 toVectorData(float3 value) {
    return value;
}

template <>
inline float3 toVectorData(float2 value) {
    return float3(value.x,value.y,0);
}

template <>
inline float3 toVectorData(color_uchar value) {
    return float3(value.red,value.green,value.blue);
}

class IntensityTransformation {
private:
	TransformationType type;
	void(*func)(const void *, void *, unsigned int, unsigned int);
public:
	IntensityTransformation() { this->type = DEFAULT; };
	IntensityTransformation(void(*func)(const void *, void *, unsigned int, unsigned int)) {
		this->func = func;
		this->type = CUSTOM;
	};

	IntensityTransformation(TransformationType type) { this->type = type; };
	template <class S, class T>
	void transform(const S * from, T * to, unsigned int length, unsigned int start = 0) {
		switch(type) {
		case HOUNSEFIELD:
			// Target must be short
			if(typeid(T) != typeid(short))
				throw SIPLException("Has to be short.");

			if(typeid(S) == typeid(short)) {
				// Do a simple copy
				this->copy(from, to, length, start);
			} else if(typeid(S) == typeid(ushort)) {
				for(unsigned int i = start; i < start+length; i++) {
					to[i] = toSingleValue(from[i]) - 1024;
				}
			} else {
				throw SIPLException("Source has to be ushort or short");
			}

			break;
		case AVERAGE:
			// Check that S is a vector type
		    if(!isVectorType(from)) {
		        throw SIPLException("Cannot convert scalar image using average intensity transformation");
		    }

            this->copy(from, to, length, start);

			break;
		case CUSTOM:
			func(from,to,length,start);
			break;
		case NORMALIZED:
			if(typeid(T) != typeid(float)) {
				throw SIPLException("Target has to float");
			} else {

			// find min and max

			float min = toSingleValue(from[start]);
			float max = toSingleValue(from[start]);
			for(unsigned int i = start; i < start+length; i++) {
				if(toSingleValue(from[i]) < min)
					min = toSingleValue(from[i]);
				if(toSingleValue(from[i]) > max)
					max = toSingleValue(from[i]);
			}
			for(unsigned int i = start; i < start+length; i++) {
				to[i] = ((float)toSingleValue(from[i])-min)/(max+min);
			}
			}
			break;
		case DEFAULT:
			if(isVectorType(from) && !isVectorType(to)) {
				// Use AVERAGE instead
				this->type = AVERAGE;
				this->transform(from,to,length,start);
			} else {
				this->copy(from, to, length, start);
			}
			break;
		default:
			throw SIPLException("Invalid transformation type");
			break;
		}
	}
	template <class S, class T>
	static void copy(const S * from, T * to, unsigned int length, unsigned int start) {
		for(unsigned int i = start; i < length+start; i++) {
			to[i] = toSingleValue(from[i]);
		}
	}
	template <class T>
	static bool isVectorType(T * from) {
		const std::type_info * i = &typeid(T);
		return *i == typeid(float3) ||
				*i == typeid(float2) ||
				*i == typeid(color_uchar) ||
				*i == typeid(color_float);
	}

};

}
#endif /* INTENSITYTRANSFORMATIONS_HPP_ */
