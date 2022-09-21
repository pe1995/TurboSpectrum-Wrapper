# external
import os
from sys import argv
import shutil
import subprocess
import datetime
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay
import pickle
import glob
import time
import warnings
# local
from atmos_package import model_atmosphere
from read_nlte import read_fullNLTE_grid, find_distance_to_point,  write_departures_forTS, restoreDepartScaling

def gradient3rdOrder(f):
    for i in range(3):
        f = np.gradient(f, edge_order=2)
    return f

def in_hull(p, hull):
    """
    Is triangulation-based interpolation to this point possible?

    Parameters
    ----------
    p : dict
        point to test, contains coordinate names as keys and their values, e.g.
        p['teff'] = 5150
    hull :

    Returns
    -------
    whether point is inside the hull
    """
    return hull.find_simplex(p) >= 0


def get_all_ma_parameters(models_path, depthScaleNew, format='m1d', debug = False):
    """
    Gets a list of all available model atmopsheres and their parameters
    for interpolation later on.
    If no list is available, creates one by scanning through all available
    models in the specified input directory.

    Parameters
    ----------
    models_path : str
        input directory contatining all available model atmospheres
    depthScaleNew : array
        depth scale (e.g. TAU500nm) to be used uniformly for model
        atmospheres and departure coefficients
        required to ensure homogenious interpolation and can be provided
        in the config file
    format : str
        format of model atmosphere, options: 'm1d' for MULTI formatted input,
        'marcs' for standard MARCS format
    debug : boolean
        switch detailed print out

    Returns
    -------
    MAgrid : dict
        dictionary containing grid of model atmospheres including both
        the parameters (like Teff, log(g), etc)
        and structure (density as a function of depth, etc)
    """

    save_file = f"{models_path}/all_models_save.pkl"

    if os.path.isfile(save_file) and os.path.getsize(save_file) > 0:
        if debug:
            print(f"reading pickled grid of model atmospheres from {save_file}")
        with open(save_file, 'rb') as f:
            MAgrid = pickle.load(f)
        depthScaleNew = MAgrid['structure'][:, np.where(MAgrid['structure_keys'][0] == 'tau500')[0][0] ]
        if np.shape(depthScaleNew) != np.shape(np.unique(depthScaleNew, axis=1)):
            print(f"depth scale is not uniform in the model atmosphere grid read from {save_file}")
            print(f"try removing file {save_file} and run the code again")
            exit()
        else:
            depthScaleNew = np.array(depthScaleNew[0])
    else:
        print(f"Checking all model atmospheres under {models_path}")

        MAgrid = {
        'teff':[], 'logg':[], 'feh':[], 'vturb':[], 'file':[], 'structure':[], 'structure_keys':[], 'mass':[]\
        }

        with os.scandir(models_path) as all_files:
            for entry in all_files:
                if not entry.name.startswith('.') and entry.is_file():
                    # try:
                    file_path = models_path + entry.name
                    ma = model_atmosphere()

                    ma.read(file_path, format=format)

                    if ma.mass <= 1.0:

                        MAgrid['teff'].append(ma.teff)
                        MAgrid['logg'].append(ma.logg)
                        MAgrid['feh'].append(ma.feh)
                        MAgrid['vturb'].append(ma.vturb[0])
                        MAgrid['mass'].append(ma.mass)

                        MAgrid['file'].append(entry.name)

                        ma.temp = np.log10(ma.temp)
                        ma.ne = np.log10(ma.ne)

                        # bring all values to the same depth_scale (tau500)
                        for par in ['temp', 'ne', 'vturb']:
                            f_int = interp1d(ma.depth_scale, ma.__dict__[par], fill_value='extrapolate')
                            ma.__dict__[par] = f_int(depthScaleNew)
                        ma.depth_scale = depthScaleNew

                        MAgrid['structure'].append( np.vstack( (ma.depth_scale, ma.temp, ma.ne, ma.vturb )  ) )
                        MAgrid['structure_keys'].append( ['tau500', 'temp', 'ne', 'vturb'])

                    # except: # if it's not a model atmosphere file, or format is wrong
                    #         if debug:
                    #             print(f"Cound not read model file {entry.name} for model atmosphere")

        for k in MAgrid:
            MAgrid[k] = np.array(MAgrid[k])

        " Check if any model atmosphere was successfully read "
        if len(MAgrid['file']) == 0:
            raise Exception(f"no model atmosphere parameters were retrived from files under {models_path}.\
Try setting debug = 1 in config file. Check that expected format of model atmosphere is set correctly.")
        else:
            print(f"{len(MAgrid['file'])} model atmospheres in the grid")

        "Print UserWarnings about any NaN in parameters"
        for k in MAgrid:
            try: # check for NaNs in numeric values:
                if np.isnan(MAgrid[k]).any():
                    pos = np.where(np.isnan(MAgrid[k]))
                    for p in pos:
                        message = f"NaN in parameter {k} from model atmosphere {MAgrid['path'][p]}"
                        warnings.warn(message, UserWarning)
            except TypeError: # ignore other [non-numerical] keys, such as path, name, etc
                pass
        "Dump all in one file (only done once)"
        with open(save_file, 'wb') as f:
            pickle.dump(MAgrid, f)
    return MAgrid

def preInterpolationTests(data, interpol_coords, valueKey, dataLabel = ''):
    """
    Run multiple tests to catch possible exceptions
    that could affect the performance of the underlying
    Qnull math engine during Delaunay triangulation
    Parameters
    ----------
    data : str
        input directory contatining all available model atmospheres
    interpol_coords : array
        depth scale (e.g. TAU500nm) to be used uniformly for model
        atmospheres and departure coefficients
        required to ensure homogenious interpolation and can be provided
        in the config file
    valueKey : str
        format of model atmosphere, options: 'm1d' for MULTI formatted input,
        'marcs' for standard MARCS format
    dataLabel : boolean
        switch detailed print out

    Returns
    -------
    boolean
    """

    " Check for degenerate parameters (aka the same for all grid points) "
    for k in interpol_coords:
        if max(data[k]) == min(data[k]):
            print(f"Grid {dataLabel} is degenerate in parameter {k}")
            print(F"Values: {np.unique(data[k])}")
            return False

    " Check for repetitive points within the requested coordinates "
    test = [ data[k] for k in interpol_coords]
    if len(np.unique(test, axis=1)) != len(test):
        print(f"Grid {dataLabel} with coordinates {interpol_coords} \
has repetitive points")
        return False

    "Any coordinates correspond to the same value? e.g. [Fe/H] and A(Fe) "
    for k in interpol_coords:
        for k1 in interpol_coords:
            if k != k1:
                diff = 100 * ( np.abs( data[k] - data[k1]) ) / np.mean(np.abs( data[k] - data[k1]))
                if np.max(diff) < 5:
                    print(f"Grid {dataLabel} is only {np.max(diff)} % different \
in parameters {k} and {k1}")
                    return False

    for k in interpol_coords:
        if np.isnan(data[k]).any():
                print(f"Warning: found NaN in coordinate {k} in grid '{dataLabel}'")
    if np.isnan(data[valueKey]).any():
        print(f"Found NaN in {valueKey} array of {dataLabel} grid")
    return True


def NDinterpolateGrid(inputGrid, interpol_par, valueKey = 'structure'):
    """
    Creates the function that interpolates provided grid.
    Coordinates of the grid are normalised and normalisation vector
    is returned for future reference.

    Parameters
    ----------
    inputGrid : dict
        contains data for interpolation and its coordinates
    interpol_par : np.array
        depth scale in the model atmosphere used to solve for NLTE RT
        (e.g. TAU500nm)
    valueKey : str
        key of the inputGrid that subset contains data for interpolation,
        e.g. 'departure'

    Returns
    -------
    interp_f : scipy.interpolate.LinearNDInterpolator
        returns interpolated data
    norm_coord : dict
        contains normalisation applied to coordinates of interpolated data
        should be used to normalised the labels provided in the call to interp_f
    """

    points = []
    norm_coord = {}
    for k in interpol_par:
            points.append(inputGrid[k] / max(inputGrid[k]) )
            norm_coord.update( { k :  max(inputGrid[k])} )
    points = np.array(points).T
    values = np.array(inputGrid[valueKey])
    interp_f = LinearNDInterpolator(points, values)

    #from scipy.spatial import Delaunay
    #print('preparing triangulation...')
    #tri = Delaunay(points)

    return interp_f, norm_coord#, tri

def prepInterpolation_MA(setup):
    """
    Read grid of model atmospheres and NLTE grids of departures
    and prepare interpolating functions
    Store for future use
    """

    " Over which parameters (== coordinates) to interpolate?"
    interpolCoords = ['teff', 'logg', 'feh'] # order should match input file!
    if 'vturb' in setup.inputParams:
        interpolCoords.append('vturb')

    "Model atmosphere grid"
    if setup.debug: print("preparing model atmosphere interpolator...")
    modelAtmGrid = get_all_ma_parameters(setup.atmos_path,  setup.depthScaleNew,\
                                    format = setup.atmos_format, debug=setup.debug)
    passed  = preInterpolationTests(modelAtmGrid, interpolCoords, \
                                    valueKey='structure', dataLabel = 'model atmosphere grid' )
    if not passed:
        exit()
    interpFunction, normalisedCoord = NDinterpolateGrid(modelAtmGrid, interpolCoords, \
                                    valueKey='structure')
    """
    Create hull object to test whether each of the requested points
    are within the original grid
    Interpolation outside of hull returns NaNs, therefore skip those points
    """
    hull = Delaunay(np.array([ modelAtmGrid[k] / normalisedCoord[k] for k in interpolCoords ]).T)

    setup.interpolator['modelAtm'] = {'interpFunction' : interpFunction, \
                                    'normCoord' : normalisedCoord, \
                                    'hull': hull}
    del modelAtmGrid
    return setup, interpolCoords

def interpolateAllPoints_MA(setup):
    """
    Python parallelisation libraries can not send more than X Gb of data between processes
    To avoid that, interpolation at each requested point is done before the start of computations
    """
    if setup.debug: print(f"Interpolating to each of {setup.inputParams['count']} requested points...")

    "Model atmosphere grid"
    setup.inputParams.update({'modelAtmInterpol' : np.full(setup.inputParams['count'], None) })

    countOutsideHull = 0
    for i in range(setup.inputParams['count']):
        point = [ setup.inputParams[k][i] / setup.interpolator['modelAtm']['normCoord'][k] \
                for k in setup.interpolator['modelAtm']['normCoord'] ]
        if not in_hull(np.array(point).T, setup.interpolator['modelAtm']['hull']):
            countOutsideHull += 1
        else:
            values =  setup.interpolator['modelAtm']['interpFunction'](point)[0]
            setup.inputParams['modelAtmInterpol'][i] = values
    if countOutsideHull > 0 and setup.debug:
        print(f"{countOutsideHull}/{setup.inputParams['count']}requested \
points are outside of the model atmosphere grid.\
No computations will be done for those")
    return setup

def prepInterpolation_NLTE(setup, el, interpolCoords, rescale = False, depthScale = None):
    """
    Read grid of departure coefficients
    in nlteData 0th element is tau, 1th--Nth are departures for N levels
    """
    if setup.debug:
        print(f"reading grid {el.nlteGrid}...")

    el.nlteData = read_fullNLTE_grid(
                                el.nlteGrid, el.nlteAux, \
                                rescale=rescale, depthScale = depthScale,
                                safeMemory = setup.safeMemory
                                )
    del el.nlteData['comment'] # to avoid confusion with dict keys

    """ Scaling departure coefficients for the most efficient interpolation """

    el.nlteData['depart'] = np.log10(el.nlteData['depart']+ 1.e-20)
    pos = np.isnan(el.nlteData['depart'])
    if setup.debug and np.sum(pos) > 0:
        print(f"{np.sum(pos)} points become NaN under log10") # none should become NaN
    el.DepartScaling = np.max(np.max(el.nlteData['depart'], axis=1), axis=0)
    el.nlteData['depart'] = el.nlteData['depart'] / el.DepartScaling

    """
    If element is Fe, than [Fe/H] == A(Fe) with an offset,
    so one of the parameters needs to be excluded to avoid degeneracy
    Here we omit [Fe/H] dimension but keep A(Fe)
    """
    if len(np.unique(el.nlteData['feh'])) == len(np.unique(el.nlteData['abund'])):
        # it is probably Fe
        if el.isFe:
            interpolCoords_el = [c for c in interpolCoords if c!='feh']
            indiv_abund = np.unique(el.nlteData['abund'])
        else:
            print(f"abundance of {el.ID} is coupled to metallicity, \
but element is not Fe (for Fe A(Fe) == [Fe/H] is acceptable)")
            exit()
    elif len(np.unique(el.nlteData['abund'])) == 1 :
    # it is either H or no iteration ovr abundance was included in computations of NLTE grids
            interpolCoords_el = interpolCoords.copy()
            indiv_abund = np.unique(el.nlteData['abund'])
    else:
        interpolCoords_el = interpolCoords.copy()
        indiv_abund = np.unique(el.nlteData['abund'] - el.nlteData['feh'])

    """
    Here we use Delaunay triangulation to interpolate over
    fund. parameters like Teff, log(g), [Fe/H], etc,
    and direct linear interpolation for abundance,
    since it is regularly spaced by construction.
    This saves a lot of time.
    """
    el.interpolator = {
            'abund' : [], 'interpFunction' : [], 'normCoord' : [], 'tri':[]
    }

    """ Split the NLTE grid into chuncks of the same abundance """
    subGrids = {
            'abund':np.zeros(len(indiv_abund)), \
            'nlteData':np.empty(len(indiv_abund), dtype=dict)
    }
    for i in range(len(indiv_abund)):
        subGrids['abund'][i] = indiv_abund[i]
        if el.isFe or el.isH:
            mask = np.where( np.abs(el.nlteData['abund'] - \
                            subGrids['abund'][i]) < 1e-3)[0]
        else:
            mask = np.where( np.abs(el.nlteData['abund'] - \
                    el.nlteData['feh'] - subGrids['abund'][i]) < 1e-3)[0]
        subGrids['nlteData'][i] = {
                    k: el.nlteData[k][mask] for k in el.nlteData
        }

    """
    Run pre-interpolation tests and eventually build an interpolating function
    for each sub-grid of constant abundance
    Grid is divided into sub-grids of constant abundance to speed-up building
    Delaunay triangulation, which is very sensitive to regular spacing
    (e.g. in abundance dimension)

    Delete intermediate sub-grids
    """
    for i in range(len(subGrids['abund'])):
        ab = subGrids['abund'][i]
        passed = preInterpolationTests(subGrids['nlteData'][i], \
                                    interpolCoords_el, \
                                    valueKey='depart', \
                                    dataLabel=f"NLTE grid {el.ID}")
        if passed:
            interpFunction, normalisedCoord  = \
                NDinterpolateGrid(subGrids['nlteData'][i], \
                    interpolCoords_el,\
                    valueKey='depart')

            el.interpolator['abund'].append(ab)
            el.interpolator['interpFunction'].append(interpFunction)
            el.interpolator['normCoord'].append(normalisedCoord)
        else:
            print("Failed pre-interpolation tests, see above")
            print(f"NLTE grid: {el.ID}, A({el.ID}) = {ab}")
            exit()
    del subGrids
    return setup

def interpolateAllPoints_NLTE(setup, el):
    """
    Interpolate to each requested abundance of element (el)
    Write departure coefficients to a file
    that will be used as input to TS later
    """
    el.departFiles = np.full(setup.inputParams['count'], None)
    for i in range(len(el.abund)):
        departFile = el.departDir + \
                f"/depCoeff_{el.ID}_{el.abund[i]:.3f}_{i}.dat"
        x, y = [], []
        # TODO: introduce class for nlte grid and set exceptions if grid wasn't rescaled
        tau = setup.depthScaleNew
        for j in range(len(el.interpolator['abund'])):
            point = [ setup.inputParams[k][i] / el.interpolator['normCoord'][j][k] \
                     for k in el.interpolator['normCoord'][j] if k !='abund']
            ab = el.interpolator['abund'][j]
            departAb = el.interpolator['interpFunction'][j](point)[0]
            if not np.isnan(departAb).all():
                x.append(ab)
                y.append(departAb)
        x, y = np.array(x), np.array(y)
        """
        Now interpolate linearly along abundance axis
        If only one point is present (e.g. A(H) is always 12),
        take departure coefficient at that abundance
        """
        if len(x) >= 2:
            if not el.isFe or el.isH:
                abScale = el.abund[i] - setup.inputParams['feh'][i]
            else:
                abScale = el.abund[i]
            if abScale > min(x) and abScale < max(x):
                depart = interp1d(x, y, axis=0)(abScale)
                depart = restoreDepartScaling(depart, el)
            else:
                depart = np.nan
        elif len(x) == 1 and el.isH:
            print(f'only one point at abundandance={x} found, will accept depart coeff.')
            depart = y[0]
            depart = restoreDepartScaling(depart, el)
        else:
            print(f"Found no departure coefficients \
at A({el.ID}) = {el.abund[i]}, [Fe/H] = {setup.inputParams['feh'][i]} at i = {i}")
            depart = np.nan

        """
        Check that no non-linearities are present
        """
        nonLin = False
        if not np.isnan(depart).all():
            for ii in range(np.shape(depart)[0]):
                if (gradient3rdOrder( depart[ii] ) > 0.01).any():
                    depart = np.nan
                    nonLin = True
                    setup.inputParams['comments'][i] += f"Non-linear behaviour in the interpolated departure coefficients \
of {el.ID} found. Will be using the closest data from the grid instead of interpolated values.\n"
                    break
        if not nonLin:
            print(f'no weird behaviour encountered for {el.ID} at abund={ el.abund[i]:.2f}')
        else:
            print(f"non-linearities for {el.ID} at abund={el.abund[i]:.2f}")
        """
        If interpolation failed e.g. if the point is outside of the grid,
        find the closest point in the grid and take a departure coefficient
        for that point
        """
        if np.isnan(depart).all():
            if setup.debug:
                print(f"attempting to find the closest point the in the grid of departure coefficients")
# TODO: move the four routines below into model_atm_interpolation
            point = {}
            for k in el.interpolator['normCoord'][0]:
                point[k] = setup.inputParams[k][i]
            if 'abund' not in point:
                point['abund'] = el.abund[i]
            pos, comment = find_distance_to_point(point, el.nlteData)
            depart = el.nlteData['depart'][pos]
            depart = restoreDepartScaling(depart, el)
            tau = el.nlteData['depthScale'][pos]

            for k in el.interpolator['normCoord'][0]:
                if ( np.abs(el.nlteData[k][pos] - point[k]) / point[k] ) > 0.5:
                    for k in el.interpolator['normCoord'][0]:
                        setup.inputParams['comments'][i] += f"{k} = {el.nlteData[k][pos]}\
(off by {point[k] - el.nlteData[k][pos] }) \n"

        write_departures_forTS(departFile, tau, depart, el.abund[i])
        el.departFiles[i] = departFile
        setup.inputParams['comments'][i] += el.comment
    return setup
