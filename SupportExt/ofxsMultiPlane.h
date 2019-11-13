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
 * Helper functions to implement plug-ins that support kFnOfxImageEffectPlaneSuite v2
 * In order to use these functions the following condition must be met:
 *#if defined(OFX_EXTENSIONS_NUKE) && defined(OFX_EXTENSIONS_NATRON)

    if (OFX::fetchSuite(kFnOfxImageEffectPlaneSuite, 2) &&  // for clipGetImagePlane
        OFX::getImageEffectHostDescription()->supportsDynamicChoices && // for dynamic layer choices
        OFX::getImageEffectHostDescription()->isMultiPlanar) // for clipGetImagePlane
         ... this is ok...
 *#endif
 */

#ifndef openfx_supportext_ofxsMultiPlane_h
#define openfx_supportext_ofxsMultiPlane_h

#include <cmath>
#include <map>
#include <string>
#include <list>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMacros.h"
#ifdef OFX_EXTENSIONS_NATRON
#include "ofxNatron.h"
#endif

#define kPlaneLabelAll "All"
#define kPlaneLabelColorAlpha "Color.Alpha"
#define kPlaneLabelColorRGB "Color.RGB"
#define kPlaneLabelColorRGBA "Color.RGBA"
#define kPlaneLabelMotionBackwardPlaneName "Backward.Motion"
#define kPlaneLabelMotionForwardPlaneName "Forward.Motion"
#define kPlaneLabelDisparityLeftPlaneName "DisparityLeft.Disparity"
#define kPlaneLabelDisparityRightPlaneName "DisparityRight.Disparity"

#define kMultiPlaneParamOutputChannels kNatronOfxParamOutputChannels
#define kMultiPlaneParamOutputChannelsChoice kMultiPlaneParamOutputChannels "Choice"
#define kMultiPlaneParamOutputChannelsRefreshButton kMultiPlaneParamOutputChannels "RefreshButton"
#define kMultiPlaneParamOutputChannelsLabel "Output Layer"
#define kMultiPlaneParamOutputChannelsHint "The layer that will be written to in output"

#define kMultiPlaneParamOutputOption0 "0"
#define kMultiPlaneParamOutputOption0Hint "0 constant channel"
#define kMultiPlaneParamOutputOption1 "1"
#define kMultiPlaneParamOutputOption1Hint "1 constant channel"

namespace OFX {
namespace MultiPlane {
struct MultiPlaneEffectPrivate;
class MultiPlaneEffect
    : public OFX::ImageEffect
{
    std::auto_ptr<MultiPlaneEffectPrivate> _imp;

public:


    MultiPlaneEffect(OfxImageEffectHandle handle);

    virtual ~MultiPlaneEffect();

    /**
     * @brief Fetch a dynamic choice parameter that was declared to the factory with describeInContextAddOutputLayerChoice() or describeInContextAddChannelChoice() and associates the given clips as dependencies of the layers in the menu.
     **/
    void fetchDynamicMultiplaneChoiceParameter(const std::string& paramName, const std::vector<OFX::Clip*>& dependsClips);

    /**
     * @brief Convenience func for param depending only on a single clip
     **/
    void fetchDynamicMultiplaneChoiceParameter(const std::string& paramName,
                                               OFX::Clip* dependsClip)
    {
        std::vector<OFX::Clip*> vec(1);

        vec[0] = dependsClip;
        fetchDynamicMultiplaneChoiceParameter(paramName, vec);
    }

    /**
     * @brief Returns the layer and channel index selected by the user in the dynamic choice param.
     * @param ofxPlane Contains the plane name defined in nuke/fnOfxExtensions.h or the custom plane name defined in ofxNatron.h
     * @param ofxComponents Generally the same as ofxPlane except in the following cases:
       - for the motion vectors planes (e.g: kFnOfxImagePlaneBackwardMotionVector)
       where the compoonents are in that case kFnOfxImageComponentMotionVectors.
       - for the color plane (i.e: kFnOfxImagePlaneColour) where in that case the ofxComponents may be kOfxImageComponentAlpha or
       kOfxImageComponentRGBA or kOfxImageComponentRGB or any other "default" ofx components supported on the clip.
     *
     * If ofxPlane is empty but the function returned true that is because the choice is either kMultiPlaneParamOutputOption0 or kMultiPlaneParamOutputOption1
     * ofxComponents will have been set correctly to one of these values.
     *
     * @param channelIndexInPlane Contains the selected channel index in the layer set to ofxPlane
     * @param isCreatingAlpha If Selected layer is the colour plane (i.e: kPlaneLabelColorAlpha or kPlaneLabelColorRGB or kPlaneLabelColorRGBA)
     * and if the user selected the alpha channel (e.g: RGBA.a), but the clip pixel components are RGB, then this value will be set to true.
     **/

    bool getPlaneNeededForParam(double time,
                                const std::string& paramName,
                                OFX::Clip** clip,
                                std::string* ofxPlane,
                                std::string* ofxComponents,
                                int* channelIndexInPlane,
                                bool* isCreatingAlpha);

    /**
     * @brief Returns the layer selected by the user in the dynamic output choice.
     * @param canUseCachedComponents If true, the output clip components will be retrieved with getCachedComponentsPresent()
     * @param ofxPlane Contains the plane name defined in nuke/fnOfxExtensions.h or the custom plane name defined in ofxNatron.h
     * @param ofxComponents Generally the same as ofxPlane except in the following cases:
       - for the motion vectors planes (e.g: kFnOfxImagePlaneBackwardMotionVector)
       where the compoonents are in that case kFnOfxImageComponentMotionVectors.
       - for the color plane (i.e: kFnOfxImagePlaneColour) where in that case the ofxComponents may be kOfxImageComponentAlpha or
       kOfxImageComponentRGBA or kOfxImageComponentRGB or any other "default" ofx components supported on the clip.
     *
     * If ofxPlane is empty but the function returned true that is because the choice is either kMultiPlaneParamOutputOption0 or kMultiPlaneParamOutputOption1
     * ofxComponents will have been set correctly to one of these values.
     **/
    bool getPlaneNeededInOutput(std::string* ofxPlane,
                                std::string* ofxComponents);


    /**
     * @brief Rebuild the given choice parameter depending on the clips components present.
     * The only way to properly refresh the dynamic choice is when getClipPreferences is called.
     * If paramName is empty, all channel menus will be refreshed.
     **/
    void buildChannelMenus(const std::string& paramName = std::string(), bool mergeEntries = true, bool addChoiceAllToOutput = false);

    /**
     * @brief Returns the clip component presents that were used for this clip in the previous call to buildChannelMenus.
     * This is only valid after a call to buildChannelMenus on a choice parameter that had this clip as dependency and
     * during the same action.
     * This is a faster way than to call clip->getComponentsPresent() since the data have already been computed.
     **/
    const std::vector<std::string>& getCachedComponentsPresent(OFX::Clip* clip) const;
    enum ChangedParamRetCode
    {
        eChangedParamRetCodeNoChange,
        eChangedParamRetCodeChoiceParamChanged,
        eChangedParamRetCodeStringParamChanged,
        eChangedParamRetCodeButtonParamChanged
    };

    /**
     * @brief To be called in the changedParam action for each dynamic choice holding channels/layers info. This will synchronize the hidden string
     * parameter to reflect the value of the choice parameter (only if the reason is a user change).
     * @return Returns true if the param change was caught, false otherwise
     **/
    ChangedParamRetCode checkIfChangedParamCalledOnDynamicChoice(const std::string& paramName, const std::string& paramToCheck, OFX::InstanceChangeReason reason);

    /**
     * @brief Calls checkIfChangedParamCalledOnDynamicChoice for all choice parameters declared. This function is just here for convenience, this is the same as calling
     * checkIfChangedParamCalledOnDynamicChoice for all parameters successively.
     * @returns True if a param change was caught, false otherwise.
     **/
    bool handleChangedParamForAllDynamicChoices(const std::string& paramName, OFX::InstanceChangeReason reason);
};

namespace Utils {
/**
 * @brief Encode the given layer and channel names into a string following the specification in ofxNatron.h
 **/
std::string makeNatronCustomChannel(const std::string& layer, const std::vector<std::string>& channels);

/**
 * @brief Given the string "comp" in the format described in ofxNatron.h, extract the layer name, the paired layer (if any)
 * and the channels
 **/
void extractChannelsFromComponentString(const std::string& comp,
                                        std::string* layer,
                                        std::string* pairedLayer,         //< if disparity or motion vectors
                                        std::vector<std::string>* channels);
}         // Utils


namespace Factory {
/**
 * @brief Add a dynamic choice parameter to select the output layer (in which the plug-in will render)
 **/
OFX::ChoiceParamDescriptor* describeInContextAddOutputLayerChoice(bool addAllChoice, OFX::ImageEffectDescriptor &desc, OFX::PageParamDescriptor* page);

/**
 * @brief Add a dynamic choice parameter to select a channel among possibly different source clips
 **/
OFX::ChoiceParamDescriptor* describeInContextAddChannelChoice(OFX::ImageEffectDescriptor &desc,
                                                              OFX::PageParamDescriptor* page,
                                                              const std::vector<std::string>& clips,
                                                              const std::string& name,
                                                              const std::string& label,
                                                              const std::string& hint);

/**
 * @brief Add the standard R,G,B,A choices for the given clips.
 * @param addConstants If true, it will also add the "0" and "1" choice to the list
 **/
void addInputChannelOptionsRGBA(OFX::ChoiceParamDescriptor* param,
                                const std::vector<std::string>& clips,
                                bool addConstants);


/**
 * @brief Same as above, but for a choice param instance
 **/
void addInputChannelOptionsRGBA(const std::vector<std::string>& clips,
                                bool addConstants,
                                std::vector<std::string>* options,
                                std::vector<std::string>* optionsLabel);
}         // Factory
}     // namespace MultiPlane
} // namespace OFX


#endif /* defined(openfx_supportext_ofxsMultiPlane_h) */
