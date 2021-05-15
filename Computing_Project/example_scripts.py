# Script to make maps of Venus and a single channel and a grid of
# all the detectors
from __future__ import division
from time_map import dataset

### Directory containing venus observation:
dir = '/Users/rpkeenan/Dropbox/4_research/3.1_TIME_analysis/venus_2/'

# load the data
print 'Venus mapper: loading data...'
venus = dataset(dir)



# reudce data - all of the following could equivalently be done by
# venus.reduce_data(thresh=100, n=2, pixel=1./120)

# remove data from turnarounds
print 'Venus mapper: handling flags...'
venus.remove_obs_flag()

# identify scans and remove data not in a scan
venus.flag_scans()
venus.remove_scan_flag()
venus.remove_short_scans(thresh=100)

# filter data using a second degree polynomial
print 'Venus mapper: filtering data...'
venus.filter_scan(n=2)

# make maps with .5 arcmin pixels
print 'Venus mapper: making maps...'
venus.make_map(pixel=1./120)



# plot a maps
print 'Venus mapper: plotting...'
venus.plot_map(0,15)   # FH 0, channel 15 has a reasonably good image of Venus

# uncomment this to plot a grid - it might come out in a strange shape
# venus.detector_grid()

print 'Done'
