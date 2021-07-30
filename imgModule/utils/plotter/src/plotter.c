#include "plotter.h"

#include "gnuplot_i.h"



void plot1d(double*data,int nPoints,char *title, char *xLabel,char *yLabel,char *style){
    gnuplot_ctrl *h1;
    h1 = gnuplot_init();

    gnuplot_resetplot(h1) ;

    gnuplot_setstyle(h1, style);
    gnuplot_set_xlabel(h1, xLabel);
    gnuplot_set_ylabel(h1, yLabel);

    gnuplot_plot_x(h1, data, nPoints, title);

    gnuplot_close(h1) ;

}







