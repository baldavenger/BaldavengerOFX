/* ***** BEGIN LICENSE BLOCK *****
 * This file is part of openfx-supportext <https://github.com/devernay/openfx-supportext>,
 * Copyright (C) 2013-2017 INRIA
 *
 * openfx-supportext is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * openfx-supportext is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with openfx-supportext.  If not, see <http://www.gnu.org/licenses/gpl-2.0.html>
 * ***** END LICENSE BLOCK ***** */

/*
 * OFX Resolution helper
 */

#ifndef IO_ofxsResolution_h
#define IO_ofxsResolution_h

#define kParamFormatPCVideo        "PC_Video"
#define kParamFormatPCVideoLabel   "PC_Video 640x480"
#define kParamFormatNTSC           "NTSC"
#define kParamFormatNTSCLabel      "NTSC 720x486 0.91"
#define kParamFormatPAL            "PAL"
#define kParamFormatPALLabel       "PAL 720x576 1.09"
#define kParamFormatNTSC169        "NTSC_16:9"
#define kParamFormatNTSC169Label   "NTSC_16:9 720x486 1.21"
#define kParamFormatPAL169         "PAL_16:9"
#define kParamFormatPAL169Label    "PAL_16:9 720x576 1.46"
#define kParamFormatHD720          "HD_720"
#define kParamFormatHD720Label     "HD_720 1280x1720"
#define kParamFormatHD             "HD"
#define kParamFormatHDLabel        "HD 1920x1080"
#define kParamFormatUHD4K          "UHD_4K"
#define kParamFormatUHD4KLabel     "UHD_4K 3840x2160"
#define kParamFormat1kSuper35      "1K_Super35(full-ap)"
#define kParamFormat1kSuper35Label "1K_Super35(full-ap) 1024x778"
#define kParamFormat1kCinemascope  "1K_Cinemascope"
#define kParamFormat1kCinemascopeLabel "1K_Cinemascope 914x778 2"
#define kParamFormat2kSuper35      "2K_Super35(full-ap)"
#define kParamFormat2kSuper35Label "2K_Super35(full-ap) 2048x1556"
#define kParamFormat2kCinemascope  "2K_Cinemascope"
#define kParamFormat2kCinemascopeLabel "2K_Cinemascope 1828x1556 2"
#define kParamFormat2kDCP          "2K_DCP"
#define kParamFormat2kDCPLabel     "2K_DCP 2048x1080"
#define kParamFormat4kSuper35      "4K_Super35(full-ap)"
#define kParamFormat4kSuper35Label "4K_Super35(full-ap) 4096x3112"
#define kParamFormat4kCinemascope  "4K_Cinemascope"
#define kParamFormat4kCinemascopeLabel  "4K_Cinemascope 3656x3112 2"
#define kParamFormat4kDCP          "4K_DCP"
#define kParamFormat4kDCPLabel     "4K_DCP 4096x2160"
#define kParamFormatSquare256      "square_256"
#define kParamFormatSquare256Label "square_256 256x256"
#define kParamFormatSquare512      "square_512"
#define kParamFormatSquare512Label "square_512 512x512"
#define kParamFormatSquare1k       "square_1k"
#define kParamFormatSquare1kLabel  "square_1k 1024x1024"
#define kParamFormatSquare2k       "square_2k"
#define kParamFormatSquare2kLabel  "square_2k 2048x2048"

namespace OFX {
enum EParamFormat
{
    eParamFormatPCVideo,
    eParamFormatNTSC,
    eParamFormatPAL,
    eParamFormatNTSC169,
    eParamFormatPAL169,
    eParamFormatHD720,
    eParamFormatHD,
    eParamFormatUHD4K,
    eParamFormat1kSuper35,
    eParamFormat1kCinemascope,
    eParamFormat2kSuper35,
    eParamFormat2kCinemascope,
    eParamFormat2kDCP,
    eParamFormat4kSuper35,
    eParamFormat4kCinemascope,
    eParamFormat4kDCP,
    eParamFormatSquare256,
    eParamFormatSquare512,
    eParamFormatSquare1k,
    eParamFormatSquare2k,
    eParamFormatCount
};

inline void
getFormatResolution(const EParamFormat format,
                    int *width,
                    int *height,
                    double *par)
{
    switch (format) {
    case eParamFormatPCVideo:
        *width =  640; *height =  480; *par = 1.; break;
    case eParamFormatNTSC:
        *width =  720; *height =  486; *par = 0.91; break;
    case eParamFormatPAL:
        *width =  720; *height =  576; *par = 1.09; break;
    case eParamFormatNTSC169:
        *width =  720; *height =  486; *par = 1.21; break;
    case eParamFormatPAL169:
        *width =  720; *height =  576; *par = 1.46; break;
    case eParamFormatHD720:
        *width = 1280; *height =  720; *par = 1.; break;
    case eParamFormatHD:
        *width = 1920; *height = 1080; *par = 1.; break;
    case eParamFormatUHD4K:
        *width = 3840; *height = 2160; *par = 1.; break;
    case eParamFormat1kSuper35:
        *width = 1024; *height =  778; *par = 1.; break;
    case eParamFormat1kCinemascope:
        *width =  914; *height =  778; *par = 2.; break;
    case eParamFormat2kSuper35:
        *width = 2048; *height = 1556; *par = 1.; break;
    case eParamFormat2kCinemascope:
        *width = 1828; *height = 1556; *par = 2.; break;
    case eParamFormat2kDCP:
        *width = 2048; *height = 1080; *par = 1.; break;
    case eParamFormat4kSuper35:
        *width = 4096; *height = 3112; *par = 1.; break;
    case eParamFormat4kCinemascope:
        *width = 3656; *height = 3112; *par = 2.; break;
    case eParamFormat4kDCP:
        *width = 4096; *height = 2160; *par = 1.; break;
    case eParamFormatSquare256:
        *width =  256; *height =  256; *par = 1.; break;
    case eParamFormatSquare512:
        *width =  512; *height =  512; *par = 1.; break;
    case eParamFormatSquare1k:
        *width = 1024; *height = 1024; *par = 1.; break;
    case eParamFormatSquare2k:
        *width = 2048; *height = 2048; *par = 1.; break;
    default:
        break;
    }
}
} // namespace OFX

#endif // ifndef IO_ofxsResolution_h

