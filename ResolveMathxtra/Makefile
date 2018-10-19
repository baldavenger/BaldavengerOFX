PLUGINNAME = ResolveMathxtra
PLUGINOBJECTS = $(PLUGINNAME).o
PLUGINOBJECTSCL = $(PLUGINNAME).o
PLUGINOBJECTSCPU = $(PLUGINNAME).o
RESOURCES = OpenFX.Yo.$(PLUGINNAME).png

## Select TRUE to install directly to OFX/Plugin folder, select FALSE for usual location. ##
INSTALL_PLUGIN := TRUE

## Select TRUE to include CUDA option, select FALSE for only OpenCL. ##
ENABLECUDA := FALSE

## Select TRUE for CPU ONLY processing (removes both CUDA and OpenCL, overrides ENABLECUDA), select FALSE for no change. ##
CPU_ONLY := TRUE

ifeq ($(ENABLECUDA), FALSE)
	PLUGINOBJECTS = $(PLUGINOBJECTSCL)
endif

ifeq ($(CPU_ONLY), TRUE)
	PLUGINOBJECTS = $(PLUGINOBJECTSCPU)
endif

TOP_SRCDIR = ..
include $(TOP_SRCDIR)/Makefile.master
