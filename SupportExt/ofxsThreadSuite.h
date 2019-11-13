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
 * A plugin-side multithread suite.
 * Can be used in place of a faulty or missing host MultiThread Suite
 */

#ifndef openfx_supportext_ofxsThreadSuite_h
#define openfx_supportext_ofxsThreadSuite_h

extern "C" {
    struct OfxMultiThreadSuiteV1;
}

namespace OFX {
    namespace Private {
        /** @brief Pointer to the plugin-side threading suite, can be used to replace gThreadSuite */
        //extern OfxMultiThreadSuiteV1 *gPluginThreadSuite;
    }

    // call from PluginFactory::load() to fix the multithread suite on some hosts that do not implement it.
    // (load() is the second argument of mDeclarePluginFactory() )
    void ofxsThreadSuiteCheck();
}

#endif // openfx_supportext_ofxsThreadSuite_h
