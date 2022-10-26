import os, inspect, sys
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ['MKL_NUM_THREADS'] = "1"
#os.environ['OPENBLAS_NUM_THREADS'] = "1"
import numpy as np
from sys import argv, exit
import glob
import pickle
from scipy import interpolate
from scipy.optimize import least_squares, minimize
import sys
sys.path.append('/home/semenova/codes/ts-wrapper/TurboSpectrum-Wrapper/utility/')
from observations import readSpectrumTSwrapper, spectrum, read_observations, convolve_gauss
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
from PayneModule import restore, restoreFromNormLabels, readNN
import cProfile
import pstats
import time
import shutil, os
from IPython.display import clear_output

def normalisePayneLabels(labels, xmax, xmin):
    return (labels-xmin)/(xmax-xmin) - 0.5
def denormalisePayneLabel(labels, xmax, xmin):
    return (labels + 0.5) * (xmax - xmin) + xmin

def callNN(p0, *args):
    wavelength, obsSpec, NNdict, freeLabels, setLabels, norm, mask, quite = args
    """
     To ensure the best convergence this function needs to be called on normalised labels (in p0)
     maybe it would work withput normalisation? it would make the code so much nicer
    """
  #  setLabels[i] = (setLabels[i] - norm['min'][i] ) / ( norm['max'][i] - norm['min'][i] ) - 0.5

    setLabels = normalisePayneLabels(setLabels, norm['max'], norm['min'])
    labels = setLabels.copy()
    labels[freeLabels] = p0
    fitLab = denormalisePayneLabel(labels[freeLabels], norm['max'][freeLabels], norm['min'][freeLabels])
    offset = denormalisePayneLabel( labels[-3], norm['max'][-3], norm['min'][-3] )
    Vbroad = denormalisePayneLabel( labels[-2], norm['max'][-2], norm['min'][-2] )
    rv = denormalisePayneLabel( labels[-1], norm['max'][-1], norm['min'][-1] )
    
    flux, wvl = [], []
    for NNid, ANN in NNdict.items():
        if min(wavelength) > min(ANN['wvl']) and max(ANN['wvl']) > max(wavelength):
            f = restoreFromNormLabels(ANN['wvl'], ANN, labels[:-3])
            flux = np.hstack( (flux,  f) )
            wvl = np.hstack( (wvl,  ANN['wvl']) )
    wvl += rv
    flux = np.interp(wavelength, wvl, flux)
    spec = spectrum(wavelength, flux)

    if obsSpec.R < ANN['res']:
        spec.convolve_resolution(obsSpec.R)
        #stdR, flux = convolve_gauss(wavelength, flux, obsSpec.R, mode='res')
    if Vbroad > 0.0:
        spec.convolve_macroturbulence(Vbroad)
        #stdV, flux = convolve_gauss(wavelength, flux, Vbroad, mode='broad')
    flux = spec.flux
    flux += offset

    chi2 = np.sum( (obsSpec.flux[mask] - flux)**2 / flux )
    
    #print(fitLab)
    #fits = glob.glob('./bestFit_*.txt')
    #if len(fits) > 0:
    #    ind = [ s.split('/')[-1].split('_')[-1].replace('.txt', '') for s in fits ]
    #    i = np.max( np.array(ind).astype(int)) + 1
    #else:
    #    i = 0
    #np.savetxt(f'./bestFit_{i:.0f}.txt', np.vstack([wavelength, obsSpec.flux[mask], flux]).T )

    #correlation = np.correlate(1-flux, 1-specObs.flux[mask])[0]
    #print(f"chi^2 = ", [f'{chi2:.4f}' if chi2 > 1 else f'{chi2:.4e}'][0])
    return chi2

def fitToNeuralNetwork(obsSpec, NNdict, prior = None, quite = True, initLabels = None):
    for NNid, NN0 in NNdict.items():
        break
    freeLabels = np.full(len(NN0['labelsKeys'])+3, True)
    setLabels = np.full(len(NN0['labelsKeys'])+3, 0.0)
    if isinstance(prior, type(None)):
        pass
    else:
        if len(prior)  < len(NN0['labelsKeys']) + 3:
            for i, l in enumerate( np.hstack( (NN0['labelsKeys'], ['offset', 'vbroad', 'rv']))):
                if l in prior or l.lower in prior:
                    freeLabels[i] = False
                    try:
                        setLabels[i] = prior[l.lower()]
                    except KeyError:
                        setLabels[i] = prior[l]
        elif prior.keys() != NN0['labelsKeys']:
            print(f"Provided prior on the labels {prior} does not match labels ANN(s) were trained on: {NN0['labelsKeys']}")
            exit()
    print(f"Fitting for {np.sum(freeLabels)} free labels")

    """
    Initialise the labels if not provided
    Extra dimension is for macro-turbulence and rotation
    """

    norm = {'min' : np.hstack( [NN0['x_min'], [-0.1, 1, -1]] ), 'max': np.hstack( [NN0['x_max'], [0.1, 50, 1]] ) }
    if isinstance(initLabels, type(None)):
        initLabels = np.zeros( len(setLabels) )

    chi2mask = np.full(len(obsSpec.flux), True)


    """
    Lambda function for fitting
    """
    fitFunc = lambda wavelength, *labels : callNN( labels, *(wavelength, obsSpec,
                                                NNdict,  freeLabels, setLabels, norm, chi2mask, quite)
                                                )
    #bounds = ( norm['min'][freeLabels], norm['max'][freeLabels] )
    bounds = []
    for i in range(np.sum(freeLabels)):
        bounds.append( [-0.5, 0.5] )
    try:
        res = minimize(
                             callNN, initLabels[freeLabels], args=(obsSpec.lam[chi2mask], obsSpec,
                                          NNdict,  freeLabels, setLabels, norm, chi2mask, quite),
                                       bounds = bounds, 
                                       options = {'disp' : True, 'maxiter' : int(1e4) },
                                       method = 'Nelder-Mead', 
                                       #method = 'Powell' 
                            )
    except RuntimeError:
        return np.full(len(setLabels), np.nan), np.nan
    " restore normalised labels "
    setLabels[freeLabels] = res.x
    setLabels[freeLabels] = denormalisePayneLabel(setLabels[freeLabels], norm['max'][freeLabels], norm['min'][freeLabels])
 
    chi2 = res.fun
    #if chi2 > 1e-2:
    #     return np.full(len(setLabels), np.nan), np.nan
   
    return setLabels, chi2

def internalAccuracyFitting(nnPath, specList, solveFor=None, lam_limits = [-np.inf, np.inf]):
    print(f"Solving for {solveFor}...")

    if isinstance(nnPath, type(str)):
        nnPath = glob.glob(nnPath)
    if len(nnPath) > 0:
        print(f"found {len(specList)} observed spectra")

    NN = {}
    wvl = []
    for nnFile in nnPath:
        NNid = nnFile.split('/')[-1].replace('.npz', '').strip()
        " Make a snapshot in time in case the training is on-going and file might be over-written "
        shutil.copy(nnFile, f"{nnFile}_snap")
        nnFile = f"{nnFile}_snap"
        print(nnFile)
        NN[NNid] = readNN(nnFile, quite=True)
        if not isinstance(solveFor, type(None)):
            if solveFor not in NN[NNid]['labelsKeys']:
                print(f"No key {solveFor} in requested NN {nnPath}")
                exit()
        wvl = np.hstack( (wvl, NN[NNid]['wvl']) )

    for NN0 in NN.values():
        break

    out = {'file':[], 'chi2':[], 'vmac':[], 'vrot':[], f"diff_{solveFor}":[]}
    totS = len(specList)
    with open(f"./fittingResults_{NNid}_fitFor{solveFor}.dat", 'w') as LogResults:
        for e, obsSpecPath in enumerate(specList):
            out['file'].append(obsSpecPath)
            obsSpec = readSpectrumTSwrapper(obsSpecPath)
            obsSpec.cut(lam_limits)
            obsSpec.ID = obsSpecPath.split('/')[-1].replace('.dat', '')
            if solveFor not in obsSpec.__dict__.keys():
                print(f"No key {solveFor} in spectrum {obsSpecPath}")
                exit()

            obsSpec.cut([min(wvl), max(wvl)] )
            if np.isfinite(NN0['res']):
                f = convolve_gauss(obsSpec.lam, obsSpec.flux, NN0['res'], mode='res')
                obsSpec = spectrum(obsSpec.lam, f, res=NN0['res'])

            prior = None
            if not isinstance(solveFor, type(None)):
                prior = {}
                for l in NN0['labelsKeys']:
                    if l.lower() != solveFor.lower():
                        prior[l.lower()] = obsSpec.__dict__[l]
                prior['vbroad'] = 0.0
                prior['rv'] = 0.0

            labelsFit, _ = fitToNeuralNetwork(obsSpec, NN, prior = prior, quite=True, initLabels=None)
            print('repeating')
            labelsFit, bestFitChi2 = fitToNeuralNetwork(obsSpec, NN, prior = prior, quite=True, initLabels=labelsFit)
            for i, l in enumerate(NN0['labelsKeys']):
                if l not in out:
                    out.update({l:[]})
                out[l].append(labelsFit[i])
            out[f"diff_{solveFor}"].append( obsSpec.__dict__[solveFor] - out[solveFor][-1] )
            d =  out[f"diff_{solveFor}"][-1]

            out['chi2'].append(bestFitChi2)
            LogResults.write( f"{obsSpec.ID} " + '\t'.join(f"{l:.3f}" for l in labelsFit) + f"{bestFitChi2 : .3f} {d:.3f}\n")

            clear_output(wait=True)
            k = f'diff_{solveFor}'
            print(f"{e:.0f}/{totS:.0f}, mean difference in {solveFor} is {np.mean(out[k]):.2f} +- {np.std(out[k]):.2f}")
    for k in out.keys():
        out[k] = np.array(out[k])
    with open(f'./fittingResults_{NNid}_fitFor{solveFor}.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(f"saved results in fittingResults_{NNid}_fitFor{solveFor}.pkl")

"""
EMCEE stuff
"""
def likelihood(labels, x, y, yerr, NN, specRes):
    ANNlabels = labels[:-3]
    vbroad = labels[-3]
    rv = labels[-2]
    log_f = labels[-1]

    # for model spectra:
    rv = 0
    vbroad = 0

    x += rv

    modelFlux = restore(x, NN, ANNlabels )
    time0 = time.time()
    if vbroad > 0.0:
        modelFlux = convolve_gauss(x, modelFlux, vbroad, mode='broad')
    #print(f"convolved in {time.time() - time0:.0f} second")
    if specRes < NN['res'] :
        modelFlux = convolve_gauss(x, modelFlux, NN['res'], mode='res')

    sigma2 = yerr**2 + modelFlux**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - modelFlux) ** 2 / sigma2 + np.log(sigma2))

def prior(labels, NN):
    ANNlabels = labels[:-3]
    vbroad = labels[-3]
    rv = labels[-2]
    log_f = labels[-1]
    check = np.full(len(ANNlabels), False)
    for i in range(len(ANNlabels)):
        if NN['x_min'][i]  < ANNlabels[i] < NN['x_max'][i]:
            check[i] = True
    if -2 < rv < 2: # in AA
        check = np.hstack( [check, [True]] )
    if 0.1 < vbroad < 20:
        check = np.hstack( [check, [True]] )
    if -10.0 < log_f < 1.0:
        check = np.hstack( [check, [True]] )

    if check.all():
        return 0.0
    else:
        return -np.inf

def probability(labels, x, y, yerr, NNpath, specRes):
    NN = readNN(NNpath, quite=True)
    lp = prior(labels, NN)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return likelihood(labels, x, y, yerr, NN, specRes) + lp

def getStartingEstimation(NN, nSamples = 100):
    """

    """
    params = []
    print(f"computing {nSamples:.0f} randomly uniformly distributed points in the ANN parameter space...")
    for i in range(len(NN['labelsKeys'])):
        dist = np.random.uniform(  min(NN['labelsInput'][i]),  max(NN['labelsInput'][i]), size=nSamples )
        params.append(dist)
    params = np.array(params).T

    fluxes = np.ones( (len(params), len(NN['wvl'])) )
    for i in range(len(params)):
        fluxes[i] = restore(NN['wvl'], NN, params[i])
    return params, fluxes


def MCMCwithANN(NNpath, spec, nwalkers=32, steps=100, startingGrid = None):
    import emcee
    from multiprocessing import Pool

    NN = readNN(NNpath)

#    profiler = cProfile.Profile()
#    profiler.enable()

    w, f = spec.lam, spec.flux
    ferr = 1/spec.snr

    if not isinstance(startingGrid, type(None)):
        sampledPoints, sampledFlux = startingGrid
        chi2 = np.full(len(sampledPoints), np.nan)
        for i in range(len(sampledPoints)):
            fMod = np.interp(w, NN['wvl'], sampledFlux[i])
            chi2[i] = np.sqrt( np.sum( (f - fMod)**2 ) )
        pos = np.where(chi2 == min(chi2))[0][0]
        startingPoint = sampledPoints[pos]
    else:  startingPoint = ( NN['x_max'] + NN['x_min'] ) / 2.



    labels = [k for k in NN['labelsKeys']]
    labels.extend(['vbroad', 'rv', 'log_f'])

    startingPoint = np.hstack([startingPoint, [0.1,0,-5] ])

    for i, l in enumerate(labels):
        if l in spec.__dict__:
            print(f"{l} = {spec.__dict__[l]:.2f}, start at {startingPoint[i]:0.2f}")

    pos = np.array(startingPoint) + np.random.randn(nwalkers, len(startingPoint)) * 1e-2 * startingPoint.T

    ndim = pos.shape[1]
    with Pool(processes=nwalkers) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, probability, args=(w, f, ferr, NNpath, spec.R),
            moves = emcee.moves.walk.WalkMove(),
            pool = pool
        )
        sampler.run_mcmc(pos, steps, progress='notebook', skip_initial_state_check=True)

#    profiler.disable()
#    with open(f"./log_profiler.txt", 'w') as stream:
#       stats = pstats.Stats(profiler, stream = stream).sort_stats('cumulative')
#       stats.print_stats()
    return sampler



def fit_uves_spectra_withPrior(arg):
    nnPath, specList, ind, linemaskFile=arg
    Jofre = {
    'HD122563' : {'teff':4587, 'logg':1.61, 'vturb':1.92, 'feh':-2.64,
                  'Mg':5.296, 'Mn':2.196, 'Co':2.248, 'Ni':3.493, 'vsini':5.0, 'key':'14023168+0941090'},
    'HD220009' : {'teff':4275, 'logg':1.47, 'vturb':1.49 , 'feh': -0.74,
                  'Mg':7.303, 'Mn': 4.193, 'Co':4.216, 'Ni':5.443, 'vsini':1.0, 'key':'23202065+0522519'},
    'HD107328' : {'teff':4496, 'logg':2.09, 'vturb':1.65, 'feh':-0.33,
                  'Mg':7.571, 'Mn':4.620, 'Co':4.710, 'Ni':5.865, 'vsini':1.9,  'key':'12202074+0318445',},
    'ksi_Hya' : {'teff':5044, 'logg':2.87, 'vturb':1.40 , 'feh': 0.16,
                  'Mg':7.684, 'Mn': 5.195, 'Co':4.881, 'Ni':6.215, 'vsini':2.4, 'key':'11325994-3151279'},
    'mu_Leo' : {'teff':4474, 'logg':2.51, 'vturb':1.28 , 'feh': 0.25,
                  'Mg':8.116, 'Mn': 5.387, 'Co':5.342, 'Ni':6.504, 'vsini':5.1,  'key':'09524561+2600243'},
    }
    NNs = glob.glob(nnPath)
    print(f"found {len(NNs)} ANNs")

    if not isinstance(linemaskFile, type(None)):
        print(f"using linemask from {linemaskFile}")
        
        lims = np.loadtxt(linemaskFile, comments=';', usecols=(0,1,2), ndmin = 1)
    else:
        lims = [None]

    ANNs = {}
    for nnPath in NNs:
           NN = readNN(nnPath, quite=True)
           ANNs[nnPath.split('/')[-1]] = NN
    fOut = open(f'./fittingResults_{ind}.dat', 'w')
    fOut.write('#  ' + '   '.join(f"{k}" for k in NN['labelsKeys']) + '  offset Vbroad RV chi  SNR line \n' )
    for sp in specList:
        starID = sp.split('/')[-2]
        print(starID, sp)
        w, f, snr = np.loadtxt(sp, unpack=True, usecols=(0,1,2))
        snr = snr[0]
        spec = spectrum(w, f, res=47000)
        spec.ID = sp.split('/')[-1].replace('.asc', '')
        spec.SNR = snr

        for lim in lims:
            if isinstance(lim, type(None)):
                lim = [ np.nan, -np.inf, np.inf ]
            print(f"fitting line at {lim[0]:.2f} AA ")
            
            specObs = spec.copy()
            specObs.cut( [lim[1], lim[2]] )
            specObs.ID +=  f"{lim[0]:.2f}_AA"
             
            if len(specObs.lam) > 0:
                #prior = None
                prior = { k: Jofre[starID][k] for k in ['teff', 'logg', 'feh', 'vturb', 'Mg', 'Mn', 'Co']}
                prior['offset'] = 0.0
                labelsFit, bestFitChi2 = fitToNeuralNetwork(specObs, ANNs, prior = prior, quite=True)
                for i, k in  enumerate(NN['labelsKeys']):
                    print(f"{k} = {labelsFit[i]:.2f}")
                print(f"offset = {labelsFit[-3]:.2f}")
                print(f"Vbroad = {labelsFit[-2]:.2f}")
                print(f"RV = {labelsFit[-1]:.2f}")
                fOut.write(f"{specObs.ID}  " +  '   '.join( f"{l:.3f}" for l in labelsFit  ) + f"  {bestFitChi2:.5e}   {snr:.0f}  {lim[0]}"  + '\n')

                # save best fit flux
                specObs = spec.copy()
                specObs.ID = spec.ID
                specObs.cut( [lim[1]-2, lim[2]+2] )
                flux, wvl = [], []
                for NNid, ANN in ANNs.items():
                    if min(specObs.lam) > min(ANN['wvl']) and max(ANN['wvl']) > max(specObs.lam):
                        f = restore(ANN['wvl'], ANN, labelsFit[:-3])
                        flux = np.hstack( (flux,  f) )
                        wvl = np.hstack( (wvl,  ANN['wvl']) )
                wvl += labelsFit[-1]
                flux = np.interp(specObs.lam, wvl, flux)
                specMod = spectrum(specObs.lam, flux)
            
                if specObs.R < ANN['res']:
                    specMod.convolve_resolution(specObs.R)
                Vbroad = labelsFit[-2]
                if Vbroad > 0.0:
                    specMod.convolve_macroturbulence(Vbroad)
                    #stdV, flux = convolve_gauss(wavelength, flux, Vbroad, mode='broad')
                flux = specMod.flux + labelsFit[-3]
                del specMod
                np.savetxt(f'./bestFit_{specObs.ID.strip()}_line{lim[0]:.2f}_SNR{specObs.SNR:.0f}.txt', np.vstack([specObs.lam, specObs.flux, flux]).T )
    fOut.close()
    return f'./fittingResults_{ind}.dat'

if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra> ")
        exit()
    # profiler = cProfile.Profile()
    # profiler.enable()

    "Fit using Payne neural network"
    nnPath = argv[1]
    obsPath = argv[2]
    specList = glob.glob(obsPath)
    print(f"found {len(specList)} observed spectra")
    if len(argv) > 2:
        linemaskFile = argv[3]
    else: 
        linemaskFile = None
    if len(argv) > 3:
        ncpu = int(argv[4])
    else: 
        ncpu = 1

    ind = np.arange(len(specList))
    args = [ [nnPath, specList[i::ncpu], i, linemaskFile ] for i in range(ncpu) ]
    with Pool(processes=ncpu) as pool:
        out = pool.map(fit_uves_spectra_withPrior, args )
    with open('./fittingResults.dat', 'w') as fOut:
        for f in out:
            for l in open(f, 'r').readlines():
                fOut.write( l )
            os.remove(f)
    

