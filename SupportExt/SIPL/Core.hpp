/**
 * Simple Image Processing Library
 * Copyright Erik Smistad 2012
 * See LICENSE file for information on use 
 */

#ifndef SIPL_H_
#define SIPL_H_

#define _USE_MATH_DEFINES // windows crap
#include <cmath>

#include "Exceptions.hpp"
#include "Types.hpp"
#include "IntensityTransformations.hpp"
#ifdef USE_GTK
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#endif
#include <fstream>
#include <typeinfo>
#include <string>
#include <stdlib.h>
#include <map>

namespace SIPL {

class Visualization;

#ifdef USE_GTK
void Init();
void Quit();

void destroyWindow(GtkWidget * widget, gpointer window) ;
void quitProgram(GtkWidget * widget, gpointer window) ;
void signalDestroyWindow(GtkWidget * widget, gpointer window) ;
void saveFileSignal(GtkWidget * widget, gpointer data) ;
void saveDialog(GtkWidget * widget, gpointer image) ;
void refresh(GtkWidget * widget, gpointer data) ;
void adjustLevelAndWindow(GtkWidget * widget, gpointer data) ;
int increaseWindowCount() ;
int getWindowCount() ;
#endif

class BaseDataset {
    public:
        virtual float * getFloatData() const = 0;
        virtual float3 * getVectorFloatData() const = 0;
        virtual float getFloatData(int3 pos) const=0;
        virtual float3 getVectorFloatData(int3 pos) const=0;
        virtual int3 getSize() const = 0;
        virtual int getTotalSize() const=0;
        virtual float3 getSpacing() const=0;
        bool isVolume;
        bool isVectorType;
        float defaultLevel;
        float defaultWindow;
        friend class Visualization;
};

template <class T>
class Dataset : public BaseDataset {
    public:
        Dataset();
        ~Dataset();
        int getWidth() const;
        int getHeight() const;
        const T * getData();
        void setData(T * data);
        virtual int getTotalSize() const=0;
        virtual int3 getSize() const = 0;
		float3 getSpacing() const;
		void setSpacing(float3 spacing);
        void fill(T value);
        float * getFloatData() const;
        float getFloatData(int3 pos) const;
        float3 * getVectorFloatData() const;
        float3 getVectorFloatData(int3 pos) const;
        void setDefaultLevelWindow();
        Visualization * display();
        Visualization * display(float level, float window);
        std::string getAttribute(std::string str);
        void setAttribute(std::string key, std::string value);
    protected:
        T * data;
        int width, height;
        float3 spacing;
        std::map<std::string, std::string> attributes;
};

template <class T>
class Image : public Dataset<T> {
    public:
        Image(const char * filepath);
        Image(unsigned int width, unsigned int height);
        Image(int2 size);
        template <class U>
        Image(Image<U> * otherImage, IntensityTransformation IT = IntensityTransformation(DEFAULT));
        T get(int i) const;
        T get(int x, int y) const;
        T get(int2 pos) const;
        T * get(Region r) const;
        void set(int x, int y, T value);
        void set(int2, T value);
        void set(int i, T v);
        void set(Region region, T value);
        int3 getSize() const;
        void save(const char * filepath, const char * imageType = "jpeg");
#ifdef USE_GTK
        void pixbufToData(GtkImage * image);
#endif
        template <class U>
        Image<T> & operator=(const Image<U> &otherImage);
        bool inBounds(int x, int y) const;
        bool inBounds(int i) const;
        int getTotalSize() const;
        Image<T> crop(Region r) const;
};

template <class T>
class Volume : public Dataset<T> {
    public:
        Volume(std::string filename, IntensityTransformation IT = IntensityTransformation(DEFAULT)); // for reading mhd files
        Volume(const char * filename, int width, int height, int depth); // for reading raw files
        Volume(int width, int height, int depth);
        Volume(int3 size);
        template <class U>
        Volume(Volume<U> * otherVolume, IntensityTransformation IT = IntensityTransformation(DEFAULT));
        T get(int x, int y, int z) const;
        T get(int3 pos) const;
        T get(int i) const;
        T * get(Region r) const;
        void set(int x, int y, int z, T value);
        void set(int3 pos, T v);
        void set(int i, T v);
        void set(Region region, T value);
        int getDepth() const;
        int3 getSize() const;
        void save(const char * filepath);
        void saveSlice(int slice, slice_plane direction, const char * filepath, const char * imageType);
        template <class U>
        Volume<T> & operator=(const Volume<U> &otherVolume);
        bool inBounds(int x, int y, int z) const;
        bool inBounds(int3 pos) const;
        bool inBounds(int i) const;
        int getTotalSize() const;
		Visualization * display();
		Visualization * display(float level, float window);
		Visualization * display(int slice, slice_plane direction);
		Visualization * display(int slice, slice_plane direction, float level, float window);
		Visualization * displayMIP();
		Visualization * displayMIP(float level, float window);
		Visualization * displayMIP(slice_plane direction);
		Visualization * displayMIP(slice_plane direction, float level, float window);
		Volume<T> * crop(Region r) const;
    private:
        int depth;
};

inline double round( double d ) {
    return floor( d + 0.5 );
}

template <class T>
void Dataset<T>::setDefaultLevelWindow() {
    this->defaultLevel = 0.5;
    this->defaultWindow = 1.0;
}


template <class T>
uchar levelWindow(T value, float level, float window) {
    float result;
    if(value < level-window*0.5f) {
        result = 0.0f;
    } else if(value > level+window*0.5f) {
        result = 1.0f;
    } else {
        result = (float)(value-(level-window*0.5f)) / window;
    }
    result = round(result*255);
    return result;
}

/* --- Spesialized toT functions --- */

void toT(bool * r, uchar * p) ;
void toT(uchar * r, uchar * p) ;
void toT(char * r, uchar * p) ;
void toT(ushort * r, uchar * p) ;
void toT(short * r, uchar * p) ;
void toT(uint * r, uchar * p) ;
void toT(int * r, uchar * p) ;
void toT(float * r, uchar * p) ;
void toT(color_uchar * r, uchar * p) ;
void toT(color_float * r, uchar * p) ;
void toT(float2 * r, uchar * p) ;
void toT(float3 * r, uchar * p) ;

#ifdef USE_GTK
template <class T>
void Image<T>::pixbufToData(GtkImage * image) {
	gdk_threads_enter ();
    GdkPixbuf * pixBuf = gtk_image_get_pixbuf((GtkImage *) image);
    for(int i = 0; i < this->width*this->height; i++) {
        guchar * pixels = gdk_pixbuf_get_pixels(pixBuf);
        unsigned char * c = (unsigned char *)((pixels + i * gdk_pixbuf_get_n_channels(pixBuf)));
        toT(&this->data[i], c);
    }
    gdk_threads_leave();
}
#endif
int validateSlice(int slice, slice_plane direction, int3 size);

template <class T>
T maximum(T a, T b, bool * change) {
    *change = a < b;
    return a > b ? a : b;
}

template <>
inline float2 maximum<float2>(float2 a, float2 b, bool * change) {
    float2 c;
    c.x = a.x > b.x ? a.x : b.x;
    c.y = a.y > b.y ? a.y : b.y;
    *change = a.x < b.x || a.y < b.y;
    return c;
}

template <>
inline float3 maximum<float3>(float3 a, float3 b, bool * change) {
    float3 c;
    c.x = a.x > b.x ? a.x : b.x;
    c.y = a.y > b.y ? a.y : b.y;
    c.z = a.z > b.z ? a.z : b.z;
    *change = a.x < b.x || a.y < b.y || a.z < b.z;
    return c;
}

template <>
inline color_uchar maximum<color_uchar>(color_uchar a, color_uchar b, bool * change) {
    color_uchar c;
    c.red = a.red > b.red ? a.red : b.red;
    c.green = a.green > b.green ? a.green : b.green;
    c.blue = a.blue > b.blue ? a.blue : b.blue;
    *change = a.red < b.red || a.green < b.green || a.blue < b.blue;
    return c;
}

/* --- Constructors & destructors --- */
template <class T>
Dataset<T>::Dataset() {
    T * d = NULL;
    this->isVectorType = IntensityTransformation::isVectorType(d);
    this->setDefaultLevelWindow();
#ifdef USE_GTK
    Init();
#endif
}

template <class T>
Image<T>::Image(const char * filename) {
#ifdef USE_GTK
    // Check if file exists
    FILE * file = fopen(filename, "r");
    this->isVolume = false;
    if(file == NULL) 
        throw FileNotFoundException(filename, __LINE__, __FILE__);
    fclose(file);
	gdk_threads_enter ();
	GtkWidget * image = gtk_image_new_from_file(filename);
	gdk_threads_leave ();
	this->height = gdk_pixbuf_get_height(gtk_image_get_pixbuf((GtkImage *) image));
	this->width = gdk_pixbuf_get_width(gtk_image_get_pixbuf((GtkImage *) image));
    this->data = new T[this->height*this->width];
    this->pixbufToData((GtkImage *)image);
    this->spacing = float3(1.0f,1.0f,1.0f);
#else
    throw SIPLCompiledWithoutGTKException(__LINE__, __FILE__);
#endif
}

template <class T> 
template <class U>
Image<T>::Image(Image<U> * otherImage, IntensityTransformation it) {
    this->isVolume = false;
    this->width = otherImage->getWidth();
    this->height = otherImage->getHeight();
    this->data = new T[this->height*this->width];
    this->spacing = otherImage->getSpacing();
    it.transform(otherImage->getData(), this->data, this->getTotalSize());
}

template <class T> 
template <class U>
Image<T>& Image<T>::operator=(const Image<U> &otherImage) {
    if(this->width != otherImage.getWidth() || this->height != otherImage.getHeight())
        throw ConversionException("image size mismatch in assignment", __LINE__, __FILE__);
    
    IntensityTransformation it;
    it.transform(otherImage->getData(), this->data, this->getTotalSize());

    return *this;
}


template <class T> 
template <class U>
Volume<T>::Volume(Volume<U> * otherImage, IntensityTransformation it) {
    this->width = otherImage->getWidth();
    this->isVolume = true;
    this->height = otherImage->getHeight();
    this->depth = otherImage->getDepth();
    this->spacing = otherImage->getSpacing();
    this->data = new T[this->height*this->width*this->depth];
    it.transform(otherImage->getData(), this->data, this->getTotalSize());
}

template <class T> 
template <class U>
Volume<T>& Volume<T>::operator=(const Volume<U> &otherImage) {
    if(this->width != otherImage.getWidth() || 
        this->height != otherImage.getHeight() ||
        this->depth != otherImage.getDepth())
        throw ConversionException("volume size mismatch in assignment", __LINE__, __FILE__);
    
    IntensityTransformation it;
    it.transform(otherImage->getData(), this->data, this->getTotalSize());

    return *this;
}


template <class T>
Volume<T>::Volume(const char * filename, int width, int height, int depth) {
    // Read raw file
    this->data = new T[width*height*depth];
    FILE * file = fopen(filename, "rb");
    if(file == NULL)
        throw FileNotFoundException(filename, __LINE__, __FILE__);
    int elementsRead = fread(this->data, sizeof(T), width*height*depth, file);
    if(elementsRead != width*height*depth) 
        throw IOException(filename, __LINE__, __FILE__);
    fclose(file);
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->isVolume = true;
    this->spacing = float3(1.0f,1.0f,1.0f);
}

template <class T>
Volume<T>::Volume(std::string filename, IntensityTransformation IT) {

    // Read mhd file
    this->data = NULL;
    std::fstream mhdFile;
    mhdFile.open(filename.c_str(), std::fstream::in);
    if(!mhdFile.is_open())
        throw IOException(filename.c_str(),__LINE__,__FILE__);
    std::string line;
    std::string rawFilename;
    bool sizeFound = false, 
         rawFilenameFound = false, 
         typeFound = false, 
         dimensionsFound = false;
    std::string typeName;
    this->spacing = float3(1.0f,1.0f,1.0f);
    do{
        std::getline(mhdFile, line);
        if(!mhdFile.eof()) {
            int firstSpace = line.find(" ");
            std::string key = line.substr(0, firstSpace);
            std::string value = line.substr(firstSpace+3);
            this->setAttribute(key, value);
        }
        if(line.substr(0, 7) == "DimSize") {
            std::string sizeString = line.substr(7+3);
            std::string sizeX = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeY = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeZ = sizeString.substr(0,sizeString.find(" "));

            this->width = atoi(sizeX.c_str());
            this->height = atoi(sizeY.c_str());
            this->depth = atoi(sizeZ.c_str());

            sizeFound = true;
        } else if(line.substr(0, 15) == "ElementDataFile") {
            rawFilename = line.substr(15+3);
            rawFilenameFound = true;

            // Remove any trailing spaces
            int pos = rawFilename.find(" ");
            if(pos > 0)
            rawFilename = rawFilename.substr(0,pos);
            
            // Get path name
            pos = filename.rfind('/');
            if(pos > 0)
                rawFilename = filename.substr(0,pos+1) + rawFilename;
        } else if(line.substr(0, 11) == "ElementType") {
            typeFound = true;
            typeName = line.substr(11+3);
            
            // Remove any trailing spaces
            int pos = typeName.find(" ");
            if(pos > 0)
            typeName = typeName.substr(0,pos);

            if(typeName == "MET_SHORT") {
            } else if(typeName == "MET_USHORT") {
            } else if(typeName == "MET_CHAR") {
            } else if(typeName == "MET_UCHAR") {
            } else if(typeName == "MET_INT") {
            } else if(typeName == "MET_UINT") {
            } else if(typeName == "MET_FLOAT") {
            } else {
                throw IOException("Trying to read volume of unsupported data type", __LINE__, __FILE__);
            }
        } else if(line.substr(0, 5) == "NDims") {
            if(line.substr(5+3, 1) == "3") 
                dimensionsFound = true;
		} else if(line.substr(0, 14) == "ElementSpacing") {
            std::string sizeString = line.substr(14+3);
            std::string sizeX = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeY = sizeString.substr(0,sizeString.find(" "));
            sizeString = sizeString.substr(sizeString.find(" ")+1);
            std::string sizeZ = sizeString.substr(0,sizeString.find(" "));

            this->spacing.x = atof(sizeX.c_str());
            this->spacing.y = atof(sizeY.c_str());
            this->spacing.z = atof(sizeZ.c_str());
        }

    } while(!mhdFile.eof());


    mhdFile.close();
    if(!sizeFound || !rawFilenameFound || !typeFound || !dimensionsFound)
        throw IOException("Error reading the mhd file", __LINE__, __FILE__);

    this->data = new T[this->width*this->height*this->depth];
    this->isVolume = true;
    if(typeName == "MET_SHORT") {
        Volume<short> * volume = new Volume<short>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    } else if(typeName == "MET_USHORT") {
        Volume<ushort> * volume = new Volume<ushort>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    } else if(typeName == "MET_CHAR") {
        Volume<char> * volume = new Volume<char>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    } else if(typeName == "MET_UCHAR") {
        Volume<uchar> * volume = new Volume<uchar>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    } else if(typeName == "MET_INT") {
        Volume<int> * volume = new Volume<int>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    } else if(typeName == "MET_UINT") {
        Volume<uint> * volume = new Volume<uint>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    } else if(typeName == "MET_FLOAT") {
        Volume<float> * volume = new Volume<float>(rawFilename.c_str(), this->width, this->height, this->depth);
        IT.transform(volume->getData(), this->data, volume->getTotalSize(), 0);
        delete volume;
    }
}

template <class T>
Volume<T>::Volume(int width, int height, int depth) {
    this->data = new T[width*height*depth];
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->isVolume = true;
    this->spacing = float3(1.0f,1.0f,1.0f);
}

template <class T>
Volume<T>::Volume(int3 size) {
    this->data = new T[size.x*size.y*size.z];
    this->width = size.x;
    this->height = size.y;
    this->depth = size.z;
    this->isVolume = true;
    this->spacing = float3(1.0f,1.0f,1.0f);
}


template <class T>
Image<T>::Image(unsigned int width, unsigned int height) {
    this->data = new T[width*height];
    this->width = width;
    this->height = height;
    this->isVolume = false;
    this->spacing = float3(1.0f,1.0f,1.0f);
}

template <class T>
Image<T>::Image(int2 size) {
    this->data = new T[size.x*size.y];
    this->width = size.x;
    this->height = size.y;
    this->isVolume = false;
    this->spacing = float3(1.0f,1.0f,1.0f);
}


template <class T>
Dataset<T>::~Dataset() {
	delete[] this->data;
}

void saveImage(BaseDataset * d, const char * filepath, const char * imageType);
template <class T>
void Image<T>::save(const char * filepath, const char * imageType) {
#ifdef USE_GTK
    saveImage(this, filepath, imageType);
#else
    std::cout << "SIPL was compiled without GTK and thus unable to save an image to disk." << std::endl;
#endif
}
template <class T>
void Volume<T>::save(const char * filepath) {
    // This might not work for the defined struct types?
	std::string filename = filepath;
	if(filename.substr(filename.size()-3) == "mhd") {
		// Create MHD file
		std::ofstream file;
		file.open(filename.c_str());
		file << "ObjectType = Image\n";
		file << "NDims = 3\n";
		file << "DimSize = " << this->width << " " <<
				this->height << " " << this->depth << "\n";
		std::string type;
		if(typeid(T) == typeid(uchar)) {
			type = "MET_UCHAR";
		} else if(typeid(T) == typeid(char)) {
			type = "MET_CHAR";
		} else if(typeid(T) == typeid(ushort)) {
			type = "MET_USHORT";
		} else if(typeid(T) == typeid(short)) {
			type = "MET_SHORT";
		} else if(typeid(T) == typeid(uint)) {
			type = "MET_UINT";
		} else if(typeid(T) == typeid(int)) {
			type = "MET_INT";
		} else if(typeid(T) == typeid(float)) {
			type = "MET_FLOAT";
		} else {
			throw SIPL::SIPLException("Unsupported type to write mhd file.", __LINE__, __FILE__);
		}
		file << "ElementType = " << type << "\n";
		file << "ElementSpacing = " << this->spacing.x << " " <<
				this->spacing.y << " " << this->spacing.z << "\n";
		// Remove path
		int pos = filename.rfind("/");
		// set filename extension to .raw
		filename = filename.substr(0, filename.size()-3) + "raw";
		std::string filenameWithoutPath = filename.substr(pos+1);
		file << "ElementDataFile = " << filenameWithoutPath << "\n";
		file.close();
	}
    FILE * file = fopen(filename.c_str(), "wb");
    if(file == NULL)
        throw IOException(filepath, __LINE__, __FILE__);

    fwrite(this->data, sizeof(T), this->width*this->height*this->depth, file);
    fclose(file);
}

Visualization * displayVisualization(BaseDataset * d, float level, float window);
template <class T>
Visualization * Dataset<T>::display() {
    return displayVisualization(this, defaultLevel, defaultWindow);
}

template <class T>
Visualization * Dataset<T>::display(float level, float window) {
    return displayVisualization(this, level, window);
}

template <class T>
Visualization * Volume<T>::display() {
    return displayVisualization(this, this->defaultLevel, this->defaultWindow);
}
template <class T>
Visualization * Volume<T>::display(float level, float window) {
    return displayVisualization(this, level, window);
}
Visualization * displayVolumeVisualization(BaseDataset * d, int slice, slice_plane direction, float level, float window);
template <class T>
Visualization * Volume<T>::display(int slice, slice_plane direction) {
    return displayVolumeVisualization(this, slice, direction, this->defaultLevel, this->defaultWindow);
}

template <class T>
Visualization * Volume<T>::display(int slice, slice_plane direction, float level, float window) {
    return displayVolumeVisualization(this, slice, direction, level, window);
}
Visualization * displayMIPVisualization(BaseDataset * d, slice_plane direction, float level, float window);
template <class T>
Visualization * Volume<T>::displayMIP() {
    return displayMIPVisualization(this, X, this->defaultLevel, this->defaultWindow);
}
template <class T>
Visualization * Volume<T>::displayMIP(float level, float window) {
    return displayMIPVisualization(this, X, level, window);
}
template <class T>
Visualization * Volume<T>::displayMIP(slice_plane direction, float level, float window) {
    return displayMIPVisualization(this, direction, level, window);
}
template <class T>
Visualization * Volume<T>::displayMIP(slice_plane direction) {
    return displayMIPVisualization(this, direction, this->defaultLevel, this->defaultWindow);
}

#ifdef USE_GTK
struct _saveData {
	GtkWidget * fs;
	Visualization * viz;
};
#endif

template <class T>
std::string Dataset<T>::getAttribute(std::string str) {
    return this->attributes[str];
}

template <class T>
void Dataset<T>::setAttribute(std::string key, std::string value) {
    this->attributes[key] = value;
}

template <class T>
int Dataset<T>::getWidth() const {
    return this->width;
}

template <class T>
int Dataset<T>::getHeight() const {
    return this->height;
}

template <class T>
int Volume<T>::getDepth() const {
    return this->depth;
}

template <class T>
int3 Image<T>::getSize() const {
    int3 size;
    size.x = this->width;
    size.y = this->height;
    size.z = 0;
    return size;
}
template <class T>
int3 Volume<T>::getSize() const {
    int3 size;
    size.x = this->width;
    size.y = this->height;
    size.z = this->depth;
    return size;
}

template <class T>
void Dataset<T>::setData(T * data) {
    this->data = data;
}

template <class T>
const T * Dataset<T>::getData() {
    return this->data;
}

template <class T>
bool Image<T>::inBounds(int i) const {
    return i >= 0 && i < this->getTotalSize();
}

template <class T>
bool Volume<T>::inBounds(int i) const {
    return i >= 0 && i < this->getTotalSize();
}

template <class T>
bool Image<T>::inBounds(int x, int y) const {
    return x >= 0 && x < this->width && y >= 0 && y < this->height;
}

template <class T>
bool Volume<T>::inBounds(int x, int y, int z) const {
    return x >= 0 && x < this->width 
        && y >= 0 && y < this->height 
        && z >= 0 && z < this->depth;
}

template <class T>
bool Volume<T>::inBounds(int3 pos) const {
    return pos.x >= 0 && pos.x < this->width 
        && pos.y >= 0 && pos.y < this->height 
        && pos.z >= 0 && pos.z < this->depth;
}


template <class T>
int Image<T>::getTotalSize() const {
    return this->width*this->height;
}

template <class T>
int Volume<T>::getTotalSize() const {
    return this->width*this->height*this->depth;
}

template <class T>
void Image<T>::set(int x, int y, T value) {
    if(!this->inBounds(x,y))
        throw OutOfBoundsException(x, y, this->width, this->height, __LINE__, __FILE__);
    this->data[x+y*this->width] = value;
}

template <class T>
void Image<T>::set(int2 pos, T value) {
    this->set(pos.x, pos.y, value);
}

template <class T>
void Image<T>::set(int i, T value) {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height, __LINE__, __FILE__);
    this->data[i] = value;
}

template <class T>
void Volume<T>::set(int i, T value) {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height*this->depth, __LINE__, __FILE__);
    this->data[i] = value;
}

template <class T>
T Image<T>::get(int x, int y) const {
    if(!this->inBounds(x,y))
        throw OutOfBoundsException(x, y, this->width, this->height, __LINE__, __FILE__);
    return this->data[x+y*this->width];
}

template <class T>
T Image<T>::get(int2 pos) const {
    return this->get(pos.x, pos.y);
}

template <class T>
T Image<T>::get(int i) const {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height, __LINE__, __FILE__);
    return this->data[i];
}

template <class T>
T * Image<T>::get(Region r) const {
    T * res = new T[r.size.x*r.size.y];
    int counter = 0;
    for(int y = r.offset.y; y < r.size.y; y++) {
    for(int x = r.offset.x; x < r.size.x; x++) {
        res[counter] = this->get(x,y);
    }}
    return res;
}

template <class T>
void Image<T>::set(Region r, T value) {
    for(int y = r.offset.y; y < r.size.y; y++) {
    for(int x = r.offset.x; x < r.size.x; x++) {
        this->set(x,y,value);
    }}
}

template <class T>
Image<T> Image<T>::crop(Region r) const {
    Image<T> * res = new Image<T>(r.size);
    res->setData(this->get(r));
}

template <class T>
T Volume<T>::get(int i) const {
    if(!this->inBounds(i))
        throw OutOfBoundsException(i, this->width*this->height*this->depth, __LINE__, __FILE__);
    return this->data[i];
}

template <class T>
void Volume<T>::set(int x, int y, int z, T value) {
    if(!this->inBounds(x,y,z))
        throw OutOfBoundsException(x, y, z, this->width, this->height, this->depth, __LINE__, __FILE__);
    this->data[x+y*this->width+z*this->width*this->height] = value;
}

template <class T>
void Volume<T>::set(int3 pos, T value) {
    this->set(pos.x, pos.y, pos.z, value);
}

template <class T>
T Volume<T>::get(int x, int y, int z) const {
    if(!this->inBounds(x,y,z))
        throw OutOfBoundsException(x, y, z, this->width, this->height, this->depth, __LINE__, __FILE__);
    return this->data[x+y*this->width+z*this->width*this->height];
}

template <class T>
T Volume<T>::get(int3 pos) const {
    return this->get(pos.x, pos.y, pos.z);
}

template <class T>
T * Volume<T>::get(Region r) const {
    T * res = new T[r.size.x*r.size.y*r.size.z];
    int counter = 0;
    for(int z = r.offset.z; z < r.offset.z+r.size.z; z++) {
    for(int y = r.offset.y; y < r.offset.y+r.size.y; y++) {
    for(int x = r.offset.x; x < r.offset.x+r.size.x; x++) {
        res[counter] = this->get(x,y,z);
        counter++;
    }}}
    return res;
}

template <class T>
void Volume<T>::set(Region r, T value) {
    for(int z = r.offset.z; z < r.offset.z+r.size.z; z++) {
    for(int y = r.offset.y; y < r.offset.y+r.size.y; y++) {
    for(int x = r.offset.x; x < r.offset.x+r.size.x; x++) {
        this->set(x,y,z,value);
    }}}
}

template <class T>
Volume<T> * Volume<T>::crop(Region r) const {
    Volume<T> * res = new Volume<T>(r.size);
    res->setData(this->get(r));
    return res;
}

template <class T>
void Dataset<T>::fill(T value) {
    for(int i = 0; i < getTotalSize(); i++)
        data[i] = value;
}

template <class T>
float3 Dataset<T>::getSpacing() const {
	return spacing;
}

template <class T>
void Dataset<T>::setSpacing(float3 spacing) {
	this->spacing = spacing;
}

template <class T>
float * Dataset<T>::getFloatData() const {
    float * floatData = new float[this->getTotalSize()];
#pragma omp parallel for
    for(int i = 0; i < this->getTotalSize(); i++) {
        floatData[i] = (float)toSingleValue(this->data[i]);
    }
    return floatData;
}

template <class T>
float Dataset<T>::getFloatData(int3 pos) const {
    return toSingleValue(this->data[pos.x+pos.y*this->width+pos.z*this->width*this->height]);
}

template <class T>
float3 Dataset<T>::getVectorFloatData(int3 pos) const {
    return toVectorData(this->data[pos.x+pos.y*this->width+pos.z*this->width*this->height]);
}


template <class T>
float3 * Dataset<T>::getVectorFloatData() const {
    float3 * floatData = new float3[this->getTotalSize()];
#pragma omp parallel for
    for(int i = 0; i < this->getTotalSize(); i++) {
        floatData[i] = toVectorData(this->data[i]);
    }
    return floatData;
}


}

 // End SIPL namespace
#endif /* SIPL_H_ */
