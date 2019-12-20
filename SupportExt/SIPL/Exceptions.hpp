/**
 * Simple Image Processing Library
 * Copyright Erik Smistad 2012
 * See LICENSE file for information on use 
 */

#ifndef SIPL_EXCEPTIONS
#define SIPL_EXCEPTIONS

#include <exception>
#include <stdio.h>
namespace SIPL {

class SIPLException : public std::exception {
    public:
        SIPLException() {
            this->line = -1;
        };
        SIPLException(const char * message) {
            this->line = -1;
            this->message = message;
        };
        SIPLException(int line, const char * file) {
            this->line = line;
            this->file = file;
        };
        SIPLException(const char * message, int line, const char * file) {
            this->message = message;
            this->line = line;
            this->file = file;
        };
        virtual const char * what() const throw() {
            char * string = new char[255];
            if(line > -1) {
                sprintf(string, "%s \nException thrown at line %d in file %s", message, line, file);
                return string;
            } else {
                return message;
            }
        };
        void setLine(int line) {
            this->line = line;
        };
        void setFile(const char * file) {
            this->file = file;
        };
        void setMessage(const char * message) {
            this->message = message;
        };
    private:
        int line;
        const char * file;
        const char * message;
};

class IOException : public SIPLException {
    public:
        IOException() {
        };
        IOException(const char * filename, int line, const char * file) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "IO Error with file %s", filename);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };
        IOException(const char * filename) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "IO Error with file %s", filename);
            this->setMessage(message);
        };
    protected:
        const char * filename;
};

class FileNotFoundException : public IOException {
    public:
        FileNotFoundException(const char * filename) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "The following file was not found: %s", filename);
            this->setMessage(message);
        };
        FileNotFoundException(const char * filename, int line, const char * file) {
            this->filename = filename;
            char * message = new char[255];
            sprintf(message, "The following file was not found: %s", filename);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };
};

class OutOfBoundsException : public SIPLException {
    public:
        OutOfBoundsException(int x, int sizeX) {
            this->x = x;
            this->sizeX = sizeX;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d in image of size %d", x, sizeX);
            this->setMessage(message);
        };
        OutOfBoundsException(int x, int sizeX, int line, const char * file) {
            this->x = x;
            this->sizeX = sizeX;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d in image of size %d", x, sizeX);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };

        OutOfBoundsException(int x, int y, int sizeX, int sizeY) {
            this->x = x;
            this->y = y;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d in image of size %d, %d", x, y, sizeX, sizeY);
            this->setMessage(message);
        };
        OutOfBoundsException(int x, int y, int sizeX, int sizeY, int line, const char * file) {
            this->x = x;
            this->y = y;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d in image of size %d, %d", x, y, sizeX, sizeY);
            this->setMessage(message);
            this->setLine(line);
            this->setFile(file);
        };
        OutOfBoundsException(int x, int y, int z, int sizeX, int sizeY, int sizeZ) {
            this->x = x;
            this->y = y;
            this->y = z;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            this->sizeZ = sizeZ;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d, %d in volume of size %d, %d, %d", x, y, z, sizeX, sizeY, sizeZ);
            this->setMessage(message);
        };
        OutOfBoundsException(int x, int y, int z, int sizeX, int sizeY, int sizeZ, int line, const char * file) {
            this->x = x;
            this->y = y;
            this->y = z;
            this->sizeX = sizeX;
            this->sizeY = sizeY;
            this->sizeZ = sizeZ;
            char * message = new char[255];
            sprintf(message, "Out of bounds exception. Requested position %d, %d, %d in volume of size %d, %d, %d", x, y, z, sizeX, sizeY, sizeZ);
            this->setMessage(message);
       };
    private:
        int x, y, z; // position requested
        int sizeX, sizeY, sizeZ;
};

class SIPLCompiledWithoutGTKException : public SIPLException {
    public:
        SIPLCompiledWithoutGTKException() {
            this->setMessage("SIPL was compiled without GTK and cannot complete");
        }
        SIPLCompiledWithoutGTKException(int line, const char * file) {
            this->setMessage("SIPL was compiled without GTK and cannot complete");
            this->setLine(line);
            this->setFile(file);
        }
};

class ConversionException : public SIPLException {
    public:
        ConversionException() : SIPLException() {};
        ConversionException(const char * message) : SIPLException(message) {};
        ConversionException(int line, const char * file) : SIPLException(line, file) {};
        ConversionException(const char * message, int line, const char * file) : SIPLException(message, line, file) { };
};

}; // END NAMESPACE SIPL

#endif
