# -*- coding: utf-8 -*-
###################################################
# SHM Detect Tool                                 #
# Shmoo plot format setting                       #
###################################################

# the keyword to indicate the site infomation line
keyword_site = 'Site'  # 'DEVICE_NUMBER:' #'Site:' #'DEVICE_NUMBER:'

# the keyword to indicate the test name line
keyword_item = '<' #'Test Name'  # 'Test Name' #'_SHM:' #'TestSuite = '

# the keyword to indicate the Shmoo plot start line
keyword_start = 'Tcoef(AC Spec)'  # "Tcoef(AC Spec)" #'Tcoef(%)'

# the keyword to indicate the Shmoo plot start line, but this is not used in the tool for now
keyword_end = 'X Axis: Vcoef(DC Spec)'#'Tcoef(%)'

# the keyword to indicate the pass symbol in Shmoo plot, reg exp support
keyword_pass = '\+'  # 'P|\*' #'\+'

# the keyword to indicate the fail symbol in Shmoo plot, reg exp support
keyword_fail = '\-|E'  # '\.|#' #'\-'

# the keyword to indicate the Y-axis position in Shmoo plot
keyword_y_axis_pos = "right"  # "left"
