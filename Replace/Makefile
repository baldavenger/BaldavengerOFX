PLUGINNAME = Replace

## Enable only one ##
ENABLECUDA := TRUE
#ENABLEAMD := TRUE
#CPU_ONLY := TRUE

INSTALL_PLUGIN := TRUE

UNAME_SYSTEM := $(shell uname -s)
ifeq ($(UNAME_SYSTEM), Darwin)
OS = macOS
endif
ifeq ($(UNAME_SYSTEM), Linux)
OS = CentOS
endif

CUDA_VERSION = 10.1

PLUGINOBJECTSEXTRA = ofxsLut.o ofxsRectangleInteract.o
PLUGINOBJECTSCUDA = $(PLUGINNAME)PluginCUDA.o $(PLUGINNAME)CudaKernel.o $(PLUGINNAME)OpenCLKernel.o
PLUGINOBJECTSAMD = $(PLUGINNAME)PluginAMD.o $(PLUGINNAME)OpenCLKernel.o $(PLUGINNAME)MetalKernel.o
PLUGINOBJECTSCPU = $(PLUGINNAME)PluginCPU.o
RESOURCES = BaldavengerOFX.$(PLUGINNAME).png

ifeq ($(UNAME_SYSTEM), Linux)
PLUGINOBJECTSCUDA = $(PLUGINNAME)PluginCUDA.o $(PLUGINNAME)CudaKernel.o $(PLUGINNAME)OpenCLKernel.o
PLUGINOBJECTSAMD = $(PLUGINNAME)PluginAMD.o $(PLUGINNAME)OpenCLKernel.o
endif

ifeq ($(ENABLECUDA), TRUE)
PLUGINOBJECTS = $(PLUGINOBJECTSCUDA) $(PLUGINOBJECTSEXTRA)
endif

ifeq ($(ENABLEAMD), TRUE)
PLUGINOBJECTS = $(PLUGINOBJECTSAMD) $(PLUGINOBJECTSEXTRA)
endif

ifeq ($(CPU_ONLY), TRUE)
PLUGINOBJECTS = $(PLUGINOBJECTSCPU) $(PLUGINOBJECTSEXTRA)
endif

TOP_SRCDIR = ..
include $(TOP_SRCDIR)/Makefile.master