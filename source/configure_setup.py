import numpy as np
import os
import shutil
from sys import argv, exit
import datetime
import glob
from scipy.interpolate import interp1d
# local
from model_atm_interpolation import get_all_ma_parameters, prepInterpolation_MA, interpolateAllPoints_MA,  prepInterpolation_NLTE, interpolateAllPoints_NLTE
from atmos_package import model_atmosphere
from read_nlte import write_departures_forTS, read_departures_forTS, restoreDepartScaling
import cProfile
import pstats
from chemical_elements import ChemElement

def read_random_input_parameters(file):
    """
    Read strictly formatted input parameters
    Example of input file:
    ....
    Teff logg Vturb FeH Fe H O # Ba
    5535  2.91  0.0  -1.03  6.470  12.0   9.610 # 2.24
    7245  5.74  0.0  -0.50  7.000  12.0   8.009 # -2.2
    ....
    Input file must include Teff, logg, Vturb, and FeH

    Parameters
    ----------
    file : str
        path to the input file

    Returns
    -------
    input_par : dict
        contains parameters requested for spectral synthesis,
        both fundamental (e.g. Teff, log(g), [Fe/H], micro-turbulence)
        and individual chemical abundances

    freeParams : list
        parameters describing model atmosphere
    """
    data =[ l.split('#')[0] for l in open(file, 'r').readlines() \
                            if not (l.startswith('#') or l.strip()=='') ]
    header = data[0].replace("'","")

    freeParams = [ s.lower() for s in header.split()[:4] ]
    for k in ['teff', 'logg', 'vturb', 'feh']:
        if k not in freeParams:
            print(f"Could not find {k} in the input file {file}. \
Please check requested input.")
            exit()
            # TODO: better exceptions/error tracking
    values =  np.array( [ l.split() for l in data[1:] ] ).astype(float)
    input_par = {
                'teff':values[:, 0], 'logg':values[:, 1], 'vturb':values[:, 2],\
                'feh':values[:,3], 'elements' : {}
                }

    elIDs = header.split()[4:]
    for i in range(len(elIDs)):
        el = ChemElement(elIDs[i].capitalize())
        el.abund = values[:, i+4]
        input_par['elements'][elIDs[i]] =  el

    input_par['count'] = len(input_par['teff'])
    input_par['comments'] = np.full(input_par['count'], '', dtype='U5000')

    if 'Fe' not in input_par['elements']:
        print(f"WARNING: input contains [Fe/H], but no A(Fe).\
Setting A(Fe) to 7.5")
        el = ChemElement('Fe')
        el.abund = input_par['feh'] + 7.5
        input_par['elements'][el.ID] = el

    absAbundCheck = np.array([ el.abund / 12. for el in input_par['elements'].values() ])
    if (absAbundCheck < 0.0).any():
        print(f"Warning: abundances must be supplied relative to H, \
on log12 scale. Please double check input file '{file}'")
    # TODO: move free parameters as a sub-dictinary of the return
    return input_par, freeParams


class setup(object):
    """
    Describes the setup requested for computations

    Parameters
    ----------
    file : str
        path to the configuration file, './config.txt' by default

    """
    def __init__(self, file='./config.txt', mode='MAinterpolate'):
        if 'cwd' not in self.__dict__.keys():
            self.cwd = f"{os.getcwd()}/"
        self.debug = 0
        self.ncpu  = 1
        self.nlte = 0
        self.safeMemory = 250 # Gb

        supportModes = ['MAinterpolate', 'MAprovided']
        if mode not in supportModes:
            print(f"Unknown mode in setup(): supported options are {supportModes}")
            exit()

        self.read_config_file(file)

        """ Any element to be treated in NLTE eventually? """
        if 'inputParams_file' in self.__dict__:
            for el in self.inputParams['elements'].values():
                if el.nlte:
                    self.nlte = True
                    break

        if 'nlte_config' not in self.__dict__ or not self.nlte:
            print(f"{50*'*'}\n Note: all elements will be computed in LTE!\n \
To set up NLTE, use 'nlte_config' flag\n {50*'*'}")

        """ Create a directory to save spectra"""
        # TODO: all directory creation should be done at the same time
        today = datetime.date.today().strftime("%b-%d-%Y")
        self.spectraDir = self.cwd + f"/spectra-{today}/"
        if not os.path.isdir(self.spectraDir):
            os.mkdir(self.spectraDir)

        if self.nlte:
            "Temporary directories for NLTE files"
            for el in self.inputParams['elements'].values():
                if el.nlte:
                    el.departDir = self.cwd + f"/{el.ID}_nlteDepFiles/"
                    if not  os.path.isdir(el.departDir):
                        os.mkdir(el.departDir)

            "TS needs to access model atoms from the same path for all elements"
            if 'modelAtomsPath' not in self.__dict__.keys():
                self.modelAtomsPath = f"{self.cwd}/modelAtoms_links/"
                if os.path.isdir(self.modelAtomsPath):
                    shutil.rmtree(self.modelAtomsPath)
                os.mkdir(self.modelAtomsPath)

                "Link provided model atoms to this directory"
                for el in self.inputParams['elements'].values():
                    if el.nlte:
                        dst = self.modelAtomsPath + el.modelAtom.split('/')[-1]
                        os.symlink(el.modelAtom, dst )

        if mode.strip() == 'MAinterpolate':
            """
            All model atmospheres should exist on the same depth scale --
            here, tau500 -- for correct interpolation
            """
            if 'depthScale' not in self.__dict__:
                self.depthScaleNew = np.linspace(-5, 2, 60)
            else: self.depthScaleNew = np.array(depthScale[0], depthScale[1], depthScale[2])

            self.interpolate()

        elif mode.strip() == 'MAprovided':
            if 'atmos_path' not in self.__dict__ or 'atmos_list' not in self.__dict__:
                print("Provide path to model atmospheres 'atmos_path' \
and path to file listing requested model atmospheres 'atmos_list' \
in the config file ")
                exit()
            self.atmos_list = np.loadtxt(self.atmos_list, ndmin=1, dtype=str)
            self.atmos_list = np.array([ self.atmos_path + f.replace('./', self.cwd) if f.startswith('./') \
                                    else self.atmos_path + f  for f in self.atmos_list])

            if self.debug:
                print(f"Requested {len(self.atmos_list):.0f} model atmospheres")
            if 'atmos_format' not in self.__dict__:
                print("Provide one of the following format keys \
in the config file as 'atmos_format' ")
                exit()

        "Some formatting required by TS routines"
        self.createTSinputFlags()


    def interpolate(self):
        """
        Here we interpolate grids of model atmospheres and
        grids of NLTE departures to each requested point
        and prepare all the files (incl. writing) in advance,
        i.e. before starting spectral computations with TS

        This decision was made to ensure that interpolation will not
        cause troubles in the middle of large expensive computations (weeks-long)
        such as computing hundreds of thousands of model spectra for surveys
        like 4MOST and WEAVE
        """
        self.interpolator = {}

        """
        Read grid of model atmospheres, conduct checks
        and eventually interpolate to every requsted point

        Model atmospheres are stored in the memory at this point
        (self.inputParams['modelAtmInterpol'])
        in contrast to NLTE departure coefficients,
        which are significantly larger and more
        NLTE departure coefficients are written to files as expected by TS
        right after interpolation
        """
        self, interpolCoords = prepInterpolation_MA(self)
        self = interpolateAllPoints_MA(self)
        del self.interpolator['modelAtm']


        """
        Go over each NLTE departure grids
        (one at a time to avoid memory overflow),
        read, conduct checks and eventually interpolate to every requested point

        Each set of departure coefficients is written in the file at this point,
        since storing all of them in the memory is not possible
        These files serve as input to TS 'as is' later anyways.

        Departure coefficients are rescaled to the same depth scale \
        as in each model atmosphere
        """
        for elID, el in self.inputParams['elements'].items():
            if el.nlte:
                print(el.ID)
                search = np.full(self.inputParams['count'], False)
                el.departFiles = np.full(self.inputParams['count'], None)
                for i in range(len(el.abund)):
                    departFile = el.departDir + \
                            f"/depCoeff_{el.ID}_{el.abund[i]:.3f}_{i}.dat"
                    el.departFiles[i] = departFile
                    if os.path.isfile(departFile):
                        search[i] = True

                if np.array(search).all():
                    print(f"Will re-use interpolated departure coefficients found under {el.departDir}")
                else:
                    self = prepInterpolation_NLTE(self, el, interpolCoords, \
                        rescale = True, depthScale = self.depthScaleNew)
                    self = interpolateAllPoints_NLTE(self, el)
                    del el.nlteData
                    del el.interpolator

    def createTSinputFlags(self):
        self.ts_input = { 'PURE-LTE':'.false.', 'MARCS-FILE':'.false.', 'NLTE':'.false.',\
        'NLTEINFOFILE':'', 'LAMBDA_MIN':4000, 'LAMBDA_MAX':9000, 'LAMBDA_STEP':0.05,\
         'MODELOPAC':'./OPAC', 'RESULTFILE':'' }


        """ At what wavelenght range to compute a spectrum? """
        self.lam_start, self.lam_end = min(self.lam_end, self.lam_start), \
                                    max(self.lam_end, self.lam_start)
        self.wave_step = np.mean([self.lam_start, self.lam_end]) / self.resolution
        self.ts_input['LAMBDA_MIN'] = self.lam_start
        self.ts_input['LAMBDA_MAX'] = self.lam_end
        self.ts_input['LAMBDA_STEP'] = self.wave_step
        self.ts_input['ts_root'] = self.ts_root


        """ Linelists """
        if type(self.linelist) == np.array or type(self.linelist) == np.ndarray:
            pass
        elif type(self.linelist) == str:
            self.linelist = np.array([self.linelist])
        else:
            print(f"Can not understand the 'linelist' flag: {self.linelist}")
            exit(1)
        llFormatted = []
        for path in self.linelist:
            if '*' in path:
                llFormatted.extend( glob.glob(path) )
            else:
                llFormatted.append(path)
        self.linelist = llFormatted
        if self.debug:
            print(f"Linelist(s) will be read from: {' ; '.join(str(x) for x in self.linelist)}")

        self.ts_input['NFILES'] = len(self.linelist)
        self.ts_input['LINELIST'] = '\n'.join(self.linelist)


        "Any element in NLTE?"
        if self.nlte:
            self.ts_input['NLTE'] = '.true.'

    def read_config_file(self, file):
        "Read all the keys from the config file"
        for line in open(file, 'r').readlines():
            line = line.strip()
            if not line.startswith('#') and len(line)>0:
                if not '+=' in line:
                    k, val = line.split('=')
                    k, val = k.strip(), val.strip()
                    if val.startswith("'") or val.startswith('"'):
                        self.__dict__[k] = val[1:-1]
                    elif val.startswith("["):
                        if '[' in val[1:]:
                            if not k in self.__dict__ or len(self.__dict__[k]) == 0:
                                self.__dict__[k] = []
                            self.__dict__[k].append(val)
                        else:
                            self.__dict__[k] = eval('np.array(' + val + ')')
                    elif '.' in val:
                        self.__dict__[k] = float(val)
                    else:
                        self.__dict__[k] = int(val)
                elif '+=' in line:
                    k, val = line.split('+=')
                    k, val = k.strip(), val.strip()
                    if len(self.__dict__[k]) == 0:
                        self.__dict__[k] = []
                    self.__dict__[k].append(val)

        if 'inputParams_file' in self.__dict__:
            self.inputParams, self.freeInputParams =\
             read_random_input_parameters(self.inputParams_file)

        if 'nlte_config' in self.__dict__:
            for l in self.nlte_config:
                l = l.replace('[','').replace(']','').replace("'","")
                elID, files = l.split(':')[0].strip().capitalize(),\
                                [f.strip() for f in l.split(':')[-1].split(',')]
                if 'nlte_grids_path' in self.__dict__:
                    files = [ f"{self.nlte_grids_path.strip()}/{f}" for f in files]
                files = [ f.replace('./', self.cwd) if f.startswith('./') \
                                                    else f  for f in files]

                if (elID not in self.inputParams['elements']) and self.debug:
                    print(f"NLTE data is provided for {elID}, \
    but it is not a free parameter in the input file {self.inputParams_file}.")
                else:
                    el = self.inputParams['elements'][elID]
                    el.nlte = True
                    el.nlteGrid = files[0]
                    el.nlteAux = files[1]
                    el.modelAtom = files[2]
