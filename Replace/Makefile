PLUGINNAME = Replace
PLUGINOBJECTSEXTRA = ofxsLut.o ofxsRectangleInteract.o
PLUGINOBJECTS = $(PLUGINNAME)Plugin.o $(PLUGINOBJECTSEXTRA) $(PLUGINNAME)CudaKernel.o $(PLUGINNAME)OpenCLKernel.o
PLUGINOBJECTSCL = $(PLUGINNAME)PluginCL.o $(PLUGINNAME)OpenCLKernel.o
PLUGINOBJECTSCPU = $(PLUGINNAME)PluginCPU.o
RESOURCES = OpenFX.Yo.$(PLUGINNAME).png

## Select TRUE to install directly to OFX/Plugin folder, select FALSE for usual location. ##
INSTALL_PLUGIN := TRUE

## Select TRUE to include CUDA option, select FALSE for only OpenCL. ##
ENABLECUDA := TRUE

## Select TRUE for CPU ONLY processing (removes both CUDA and OpenCL, overrides ENABLECUDA), select FALSE for no change. ##
CPU_ONLY := FALSE

ifeq ($(ENABLECUDA), FALSE)
	PLUGINOBJECTS = $(PLUGINOBJECTSCL) $(PLUGINOBJECTSEXTRA)
endif

ifeq ($(CPU_ONLY), TRUE)
	PLUGINOBJECTS = $(PLUGINOBJECTSCPU) $(PLUGINOBJECTSEXTRA)
endif

TOP_SRCDIR = ..
include $(TOP_SRCDIR)/Makefile.master
