#ifndef VISUALIZATION_HPP_
#define VISUALIZATION_HPP_

#include "Core.hpp"
#include <vector>
#include <map>
#include <string>
#ifdef USE_GTK
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gdk/gdkkeysyms.h>
#endif

namespace SIPL {

enum visualizationType {
    SLICE, MIP
};

class BaseDataset;

class Visualization {
    public:
        Visualization(BaseDataset * image);
        Visualization(BaseDataset * image, BaseDataset * image2);
        Visualization(BaseDataset * image, BaseDataset * image2, BaseDataset * image3);
        void setLevel(float level);
        float getLevel(BaseDataset * image);
        void setWindow(float window);
        void setLevel(BaseDataset * image, float level);
        void setWindow(BaseDataset * image, float window);
        float getWindow(BaseDataset * image);
        void setTitle(std::string);
        void setType(visualizationType);
        void setScale(float scale);
        void addImage();
        void addVolume();
        void display();
        void update();
        void draw();
#ifdef USE_GTK
        static void keyPressed(GtkWidget * widget, GdkEventKey * event, gpointer user_data);
        static bool buttonPressed(GtkWidget * widget, GdkEventButton * event, gpointer user_data);
#endif
        slice_plane getDirection() const;
        void setDirection(slice_plane direction);
        int getSlice() const;
        void setSlice(int slice);
        int3 getSize();
        float getAngle() const;
        void setAngle(float angle);
#ifdef USE_GTK
        GdkPixbuf * render();
#endif
        std::vector<BaseDataset *> getImages();
#ifdef USE_GTK
        GtkWidget * getGtkImage();
#endif
        int getWidth();
        int getHeight();
        float getSpacingX();
        float getSpacingY();
    private:
        bool isVolumeVisualization;
        std::vector<BaseDataset *> images;
        std::map<BaseDataset *, float> level;
        std::map<BaseDataset *, float> window;
        slice_plane direction;
        int slice;
#ifdef USE_GTK
        void renderSlice(int, GdkPixbuf *);
        void renderImage(int, GdkPixbuf *);
        void renderMIP(int, GdkPixbuf *);
#endif
        std::string title;
        float scale;
        float angle;
        int width, height;
        int3 size;
#ifdef USE_GTK
        GtkWidget * gtkImage;
        GtkWidget * scaledImage;
        GtkWidget * statusBar;
#endif
        void zoomIn();
        void zoomOut();
        visualizationType type;
        float3 getValue(int2 position);
        int3 getTrue3DPosition(int2 pos);
        float spacingX;
        float spacingY;
};


}; // END SIPL NAMESPACE

#endif /* VISUALIZATION_HPP_ */
