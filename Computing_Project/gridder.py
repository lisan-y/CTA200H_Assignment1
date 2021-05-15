from __future__ import division

from numpy import amax, amin, arange, array, average, concatenate, copy, empty, indices, linspace, logspace, moveaxis, ndarray, ones, split, zeros
from numpy import array_equal, ceil, conj, delete, exp, isclose, log, log10, mean, nonzero, pi, prod, std, sum, sqrt, where, setdiff1d, expand_dims, copy, add
'''error class for function input errors
'''

class InputError(Exception):
    def __init__(self,expression,message):
        Exception.__init__ (self,'bad input in '+expression+' - '+message)

##################################################
######### SECTION 1: Put Data on a Grid ##########
##################################################
'''MK_GRID:
This function takes a list of positions and places them on a grid.
By default the grid is populated with object counts and is fit to
the position data given. Optionally, values can be given in place
of strict counting. Multiple properties can be gridded
simultaneously. A number of grid parameters can be specified.

Required arguments:
data_pos - an n x d array where n is the number of objects and d
 is the number of dimensions of the space. This should specify the
 location in d-dimensional space of each of the n objects to be
 placed on the grid.

Optional arguments:
data_value - an n x p array where n is the number of objects and
 p is the number of properties associated with each object to be
 placed gridded. The values of each column should be the properties
 values associated with the corresponding objects in data_pos.
 If this argument is not given, the function will return counts
 per voxel.
calc_density (= True) - if True it will calculate a density per
 voxel rather than a total, implemented by dividing voxel values
 by the voxel volume (product of voxel lenghts)
voxel_length (= 1) - allows the user to specify the side length of
 pixels (in units matching those of data_pos). Can either be a
 scalar or a list of length d (where d is the number of dimensions
 of the space being gridded), allowing voxels to have un-equal
 dimensions.
fit_dims (= False) - if true the function will fit the grid with
 one corner at corresponding to the minimum (by dimension) of the
 values in data_pos and extending out in increments of voxel_length
 to include all points given. If false, a grid must be specified
 using the parameters dims and center.
dims - if fit_dims is set to false this argument should be given
 to specify the side length of the grid being created. It should
 be either a scalar or a list of length d where d is the number of
 spatial dimensions of the grid.
center (= dims/2) - if fit_dims is set to false this argmunt can
 be used to specify the center of the grid being created.

Returns:
grid - an n (or n + 1) dimensional array of grid points and the
 value (values) ath those points.
axes - a list of length n, containing arrays of the grid axes
'''

def mk_grid(data_pos,
            data_value = array([]), calc_density = False,
            voxel_length = 1,
            fit_dims = True, dims = 0, center = array([])):

# fill out input data and calculate some dimensionality information
# for error checking later
    # fill in unassigned data_value to do simple counting
    if array_equal(data_value, array([])):
        data_value = ones(len(data_pos),dtype='int')

    # number of objects and properties - n = dimensionality of position
    # m = number of properties
    n = len(data_pos[0])
    if data_value.ndim > 1:
        m = len(data_value[0])
    else:
        m = 1

# look for some errors in input
    if isinstance(voxel_length, ndarray) == False:
        voxel_length = array(voxel_length)
    if voxel_length.size not in (n,1):
        raise InputError('voxel_length','voxel sizes given don\'t match position data')
    # convert position lists into array
    if isinstance(data_pos, ndarray) == False:
        data_pos = array(data_pos)

    # convert value lists into array
    if isinstance(data_value, ndarray) == False:
        data_value = array(data_value)
    if len(data_value) != len(data_pos):
        raise InputError('data_value','value array not the same length as position array')

    # calculate voxel volume
    if voxel_length.size == 1:
        voxel_length = ones(n)*voxel_length
    voxel_volume = prod(voxel_length)

# This section determines the dimensions for the grid
    # fit dimensions of grid if dimension fitting turned on
    if fit_dims:
        # calculate spread of data to create grid
        data_min = []
        data_max = []

        for i in arange(0,n):
            data_min.append(amin(data_pos.T[i]))
            data_max.append(amax(data_pos.T[i]))

        data_min = array(data_min)
        data_max = array(data_max)

        data_range = data_max-data_min

        # calculate grid size
        if m == 1:
            grid_length = ceil(data_range/voxel_length).astype(int)
        else:
            grid_length = concatenate((ceil(data_range/voxel_length).astype(int),[m]))

        # center and scale data
        data_pos_mod = data_pos - data_min
        data_pos_mod /= voxel_length

        # record center for making grid axes
        center = data_min + data_range/2

    # if dimension fitting turned off check that dimensions have been given
    else:
        # check that right arugments are given
        if dims == 0:
            raise InputError('fit_dims','if fit_dims is set to false the dimensions of the box to be fitted must be specified in dims')
        if isinstance(dims, ndarray) == False:
            dims = array(dims)
        if dims.size not in (n,1):
            raise InputError('dims','dimensions given don\'t match position data')

        if m == 1:
            grid_length = (ones(n)*ceil(dims/voxel_length)).astype(int)
        else:
            grid_length = concatenate(((ones(n)*ceil(dims/voxel_length)).astype(int),[m]))

        # center data
        if array_equal(center, array([])):
            center = dims/2
            data_pos_mod = copy(data_pos)
        elif len(center) != n:
            raise InputError('center','center position must match dimensions of data')
        else:
            data_pos_mod = data_pos - center + dims/2

        for i in arange(n):
            ind = where((data_pos_mod[:,i] < dims[i]) & (data_pos_mod[:,i] > 0))
            data_pos_mod = data_pos_mod[ind]
            data_value = data_value[ind]

        # scale data
        data_pos_mod /= voxel_length

    # assign positions
    GRID = zeros(grid_length)
    if m == 1:
        add.at(GRID,split(data_pos_mod.astype(int),n,axis=1),expand_dims(data_value,1))
    else:
        add.at(GRID,split(data_pos_mod.astype(int),n,axis=1),expand_dims(data_value,1))

    # calculate density
    if calc_density == True:
        GRID /= voxel_volume

    # record the axes of the grid
    if fit_dims:
        ax_min = data_min
    else:
        ax_min = center-dims/2
    AXES = []

    for i in arange(0,n):
        shape = ()
        for j in arange(0,n):
            if i == j:
                shape = shape + tuple([grid_length[i]])
            else:
                shape = shape + tuple([1])
        axis = array([ax_min[i] + voxel_length[i]*(k+0.5) for k in range(grid_length[i])]).reshape(shape)
        AXES.append(axis)

    return GRID, AXES
