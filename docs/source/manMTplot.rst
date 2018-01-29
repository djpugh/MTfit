********************************
MTplot command line options
********************************
Command line usage::
MTplot [-h] [-d DATAFILE] [-p PLOT_TYPE] [-c COLORMAP] [-f FONTSIZE]
              [-l LINEWIDTH] [-t] [-r RESOLUTION] [-b BINS] [--fault_plane]
              [-n] [--tnp] [--marker_size MARKERSIZE]
              [--station_marker_size STATION_MARKERSIZE] [--showmaxlikelihood]
              [--showmean] [--grid_lines] [--color COLOR] [--type_label]
              [--hex_bin] [--projection PROJECTION] [--save SAVE_FILE]
              [--save-dpi SAVE_DPI] [--version]
              [data_file]

Positional Arguments:
============================

  data_file             Data file to use for plotting, optional but must be
                        specified either as a positional argument or as an
                        optional argument (see -d below)
                         
                         

Optional Arguments:
============================

  -h, --help            show this help message and exit
                         
                         
  -d DATAFILE, --datafile DATAFILE, --data_file DATAFILE
                        MTplot can read the output data from MTfit
                         
                         
  -p PLOT_TYPE, --plot_type PLOT_TYPE, --plottype PLOT_TYPE,
  --plot-type PLOT_TYPE, --type PLOT_TYPE

                        Type of plot to make
                         
                         
  -c COLORMAP, --colormap COLORMAP, --color_map COLORMAP, --color-map COLORMAP
                        Matplotlib colormap selection
                         
                         
  -f FONTSIZE, --font_size FONTSIZE, --fontsize FONTSIZE, --font-size FONTSIZE
                        Fontsize
                         
                         
  -l LINEWIDTH, --line_width LINEWIDTH, --linewidth LINEWIDTH,
  --line-width LINEWIDTH

                        Linewidth
                         
                         
  -t, --text, --show-text, --show_text, --showtext
                        Flag to show text or not
                         
                         
  -r RESOLUTION, --resolution RESOLUTION
                        Resolution for the focal sphere plot types
                         
                         
  -b BINS, --bins BINS  Number of bins for the histogram plot types
                         
                         
  --fault_plane, --faultplane, --fault-plane
                        Show the fault planes on a focal sphere type plot
                         
                         
  -n, --nodal_line, --nodal-line, --nodalline
                        Show the nodal lines on a focal sphere type plot
                         
                         
  --tnp, --tp, --pt     Show TNP axes on focal sphere plots
                         
                         
  --marker_size MARKERSIZE, --markersize MARKERSIZE, --marker-size MARKERSIZE
                        Set marker size
                         
                         
  --station_marker_size STATION_MARKERSIZE,
  --stationmarkersize STATION_MARKERSIZE,
  --station-marker-size STATION_MARKERSIZE,
  --station_markersize STATION_MARKERSIZE, --station-markersize STATION_MARKERSIZE

                        Set station marker size
                         
                         
  --showmaxlikelihood, --show_max_likelihood, --show-max-likelihood
                        Show the maximum likelihood solution on a fault plane
                        plot (shown in color set by --color).
                         
                         
  --showmean, --show-mean, --show_mean
                        Show the mean orientaion on a fault plane plot (shown
                        in green).
                         
                         
  --grid_lines, --gridlines, --grid-lines
                        Show interior lines on Hudson and lune plots
                         
                         
  --color COLOR         Set default color
                         
                         
  --type_label, --typelabel, --type-label
                        Show source type labels on Hudson and lune plots.
                         
                         
  --hex_bin, --hexbin, --hex-bin
                        Use hex bin for histogram plottings
                         
                         
  --projection PROJECTION
                        Projection choice for focal sphere plots
                         
                         
  --save SAVE_FILE, --save_file SAVE_FILE, --savefile SAVE_FILE,
  --save-file SAVE_FILE

                        Set the filename to save to (if set the plot is saved
                        to the file)
                         
                         
  --save-dpi SAVE_DPI, --savedpi SAVE_DPI, --save_dpi SAVE_DPI
                        Output file dpi
                         
                         
  --version             show program's version number and exit
                         
                         
