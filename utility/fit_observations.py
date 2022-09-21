import numpy as np
from sys import argv, exit
import os
import glob
import pickle
from scipy import interpolate
import sys
sys.path.append('/home/semenova/codes/ts-wrapper/TurboSpectrum-Wrapper/utility/')
from observations import readSpectrumTSwrapper, spectrum, read_observations, convolve_gauss
from multiprocessing import Pool
from scipy.optimize import curve_fit, fmin_bfgs
from PayneModule import restore, restoreFromNormLabels, readNN
import cProfile
import pstats
import time
import shutil
from IPython.display import clear_output

def normalisePayneLabels(labels, xmax, xmin):
    return (labels-xmin)/(xmax-xmin) - 0.5

def callNN(wavelength, obsSpec, NNdict, p0, freeLabels, setLabels, mask, quite=True):
    """
     To ensure the best convergence this function needs to be called on normalised labels (in p0)
     maybe it would work withput normalisation? it would make the code so much nicer
    """
  #  setLabels[i] = (setLabels[i] - norm['min'][i] ) / ( norm['max'][i] - norm['min'][i] ) - 0.5    

    labels = setLabels.copy()
    labels[freeLabels] = p0
    print(labels[freeLabels])
    Vbroad = labels[-3] 
    rv = labels[-2]
    offset = labels[-1]
    flux, wvl = [], []
    for NNid, ANN in NNdict.items():
        f = restore(ANN['wvl'], ANN, labels[:-3])
        flux = np.hstack( (flux,  f) )
        wvl = np.hstack( (wvl,  ANN['wvl']) )
    wvl += rv
    if Vbroad > 0.0:
        flux = convolve_gauss(wavelength, flux, Vbroad, mode='broad')
    if obsSpec.R < ANN['res']:
        flux = convolve_gauss(wavelength, flux, obsSpec.R, mode='res')
    flux = np.interp(wavelength, wvl, flux)
    flux += offset

    chi2 = np.sqrt(np.sum(obsSpec.flux[mask] - flux)**2)
    fits = glob.glob('./bestFit_*.txt')
    if len(fits) > 0:
        ind = [ s.split('/')[-1].split('_')[-1].replace('.txt', '') for s in fits ]
        i = np.max( np.array(ind).astype(int)) + 1
    else:
        i = 0
    np.savetxt(f'./bestFit_{i:.0f}.txt', np.vstack([wavelength, obsSpec.flux[mask], flux]).T ) 
    print(f"chi^2 = {chi2:.2f}")
    return flux

def fitToNeuralNetwork(obsSpec, NNdict, prior = None, quite = True):
    for NNid, NN0 in NNdict.items():
        break
    freeLabels = np.full(len(NN0['labelsKeys'])+3, True)
    setLabels = np.full(len(NN0['labelsKeys'])+3, 0.0)
    if isinstance(prior, type(None)):
        pass
    else:
        if len(prior)  < len(NN0['labelsKeys']) + 1:
            for i, l in enumerate( np.hstack( (NN0['labelsKeys'], ['vbroad', 'rv', 'offset']))):
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

    initLabels = []
    norm = {'min' : np.hstack( [NN0['x_min'], [1, -0.3, -0.1]] ), 'max': np.hstack( [NN0['x_max'], [100, 0.3, 0.1]] ) }
    for i, l in enumerate( np.hstack( (NN0['labelsKeys'], ['vbroad', 'rv', 'offset']))):
        if freeLabels[i]:
            initLabels.append( np.mean( (norm['min'][i], norm['max'][i] ) ) )
    #"""
    #Resampled (and cut if needed)  observed spectrum to the wavelength points 
    #provided in the ANN
    #"""
    #w_new = []
    #for ann in NNdict.values():
    #    w_new = np.hstack( (w_new, ann['wvl']) )
    #w_new = w_new[np.logical_and( w_new>min(obsSpec.lam ), w_new<max(obsSpec.lam) )]
    #obsSpec.flux = np.interp(w_new,  obsSpec.lam, obsSpec.flux)
    #obsSpec.lam = w_new

    #y = np.gradient(specObs.flux)
    #chi2mask = np.where(np.abs(y) > np.std(y))
    #chi2mask = np.where(1.-specObs.flux > 3*1/np.mean(specObs.SNR)  )
    chi2mask = len(specObs.flux) * [True]
    if len(specObs.flux[chi2mask]) > 0.7 * len(specObs.flux) :
    
    
        """
        Lambda function for fitting 
        """
        fitFunc = lambda wavelength, *labels : callNN(
                                                    wavelength, obsSpec,
                                                    NNdict, labels, freeLabels, setLabels, chi2mask, quite = quite
                                                    )
        bounds = ( norm['min'][freeLabels], norm['max'][freeLabels] )
        print(f"Bounds: {bounds}")
        try:
            popt,_ = curve_fit(
                            fitFunc, obsSpec.lam[chi2mask], \
                            obsSpec.flux[chi2mask], p0=initLabels,\
                            bounds = bounds,
                            method = 'dogbox',
                            )
        except RuntimeError:
            return np.full(len(setLabels), np.nan), np.nan
        " restore normalised labels "
        setLabels[freeLabels] = popt
    
        wavelength = obsSpec.lam
        #chi2 = np.sqrt(np.sum(obsSpec.flux - flux)**2)
        chi2 = np.inf
        return setLabels, chi2
    else:
        return np.full(len(setLabels), np.nan), np.nan


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

            labelsFit, bestFitChi2 = fitToNeuralNetwork(obsSpec, NN, prior = prior, quite=True)
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
def likelihood(labels, x, y, yerr, NN):
    ANNlabels = labels[:-1]
    #ANNlabels = [5421.00,  2.72, 0.70, -1.51, 5.95, 3.88, 3.46]
    #ANNlabels.append(labels[0])
    log_f = labels[-1]
    modelFlux = restore(x, NN, ANNlabels )
    #if Vbroad > 0.0:
    #    flux = convolve_gauss(x, modelFlux, Vbroad, mode='broad')
    #if NN['res'] < np.inf:
    #    flux = convolve_gauss(x, modelFlux, NN['res'], mode='res')
    sigma2 = yerr**2 + modelFlux**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - modelFlux) ** 2 / sigma2 + np.log(sigma2))

def prior(labels, NN):
    ANNlabels = labels[:-1]
    #ANNlabels = [5421.00,  2.72, 0.70, -1.51, 5.95, 3.88, 3.46]
    #ANNlabels.append(labels[0])
    log_f = labels[-1]
    check = np.full(len(ANNlabels), False)
    for i in range(len(ANNlabels)):
        if NN['x_min'][i]  < ANNlabels[i] < NN['x_max'][i]:
            check[i] = True
    if check.all() and -10.0 < log_f < 1.0:
        return 0.0
    else:
        return -np.inf

def probability(labels, x, y, yerr, NNpath):
    NN = readNN(NNpath, quite=True)
    lp = prior(labels, NN)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return likelihood(labels, x, y, yerr, NN) + lp
    
def MCMCwithANN(NNpath, specPath):
    #import mkl
    #mkl.set_num_threads(100)

    NN = readNN(NNpath)
    spec = readSpectrumTSwrapper(specPath)  
    #spec.cut([6170, 6190])
    spec.cut([min(NN['wvl']), max(NN['wvl'])])
    computedLabels = [ spec.__dict__[k] for k  in NN['labelsKeys'] ]
    for l in NN['labelsKeys']:
        print(f"{l} = {spec.__dict__[l]:.2f}")
    w, f = spec.lam, spec.flux
    ferr = np.full(len(f), 0.01)
    import emcee

    startingPoint = ( NN['x_max'] + NN['x_min'] ) / 2.
    #startingPoint = computedLabels
    print(startingPoint)
    startingPoint = np.hstack([startingPoint, [-5]])
    # Ni
    #startingPoint[7] = startingPoint[7] - 1
    #startingPoint.append(-5)
    #startingPoint = [3, -5]
    nwalkers = 32
    pos = np.array(startingPoint) + np.random.randn(nwalkers, len(startingPoint)) * 1e-2 * startingPoint.T

    ndim = pos.shape[1]
    from multiprocessing import Pool

    with Pool(processes=nwalkers) as pool:
    
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, probability, pool = pool, args=(w, f, ferr, NNpath)
        )
        sampler.run_mcmc(pos, 100, progress=True)

    flat_samples = sampler.get_chain(discard=10, thin=1, flat=True)
    for i in range(ndim-1):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{NN['labelsKeys'][i]} = {mcmc[1]:.3f} + {q[0]:.3f} - {q[1]:.3f}")
    i = ndim-1
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    print(f"log_f = {mcmc[1]:.3f}") 
    

if __name__ == '__main__':
    if len(argv) < 3:
        print("Usage: $ python ./fit_observations.py \
<path to model spectra or payne NN> <path to observed spectra> ")
        exit()
    profiler = cProfile.Profile()
    profiler.enable()

    "Fit using Payne neural network"
    nnPath = argv[1]
    NNs = glob.glob(nnPath)
    print(f"found {len(NNs)} ANNs")
    obsPath = argv[2]
    specList = glob.glob(obsPath)
    #specList = glob.glob('./uves_test/ksi_Hya/uvu_11325994-3151279_520.0_25_reNormCont.asc')
    print(f"found {len(specList)} observed spectra")
    
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
    
    lim = None
    lim = [6176, 6178]

    ANNs = {}
    for nnPath in NNs:
           NN = readNN(nnPath, quite=True)
           ANNs[nnPath.split('/')[-1]] = NN
    with open('./fittingResults.dat', 'w') as fOut:
            fOut.write('#  ' + '   '.join(f"{k}" for k in NN['labelsKeys']) + ' Vbroad RV  offset chi  SNR\n' )
            
            #profiler = cProfile.Profile()
            #profiler.enable()
            for sp in specList:
                starID = sp.split('/')[-2]
                print(starID)
                w, f, snr = np.loadtxt(sp, unpack=True, usecols=(0,1,2))
                snr = snr[0]
                if isinstance(lim, type(None)):
                    lim = [ min(w), max(w) ]
                if  lim[0] >= min(w) and  lim[1] <= max(w):
                    print(sp)
                    specObs = spectrum(w, f, res=47000)
                    specObs.ID = sp.split('/')[-1].replace('.asc', '')
                    specObs.SNR = snr
                    specObs.cut(lim)
                    print(len(specObs.lam))
                    #prior = None
                    prior = { k: Jofre[starID][k] for k in ['teff', 'logg', 'feh', 'vturb', 'Mg', 'Mn', 'Co']}
                    prior['offset'] = 0.0
                    print(prior)
                    labelsFit, bestFitChi2 = fitToNeuralNetwork(specObs, ANNs, prior = prior, quite=True)
                    for i, k in  enumerate(NN['labelsKeys']):
                        print(f"{k} = {labelsFit[i]:.2f}")
                    print(f"Vbroad = {labelsFit[-3]:.2f}")
                    print(f"RV = {labelsFit[-2]:.2f}")
                    print(f"offset = {labelsFit[-1]:.2f}")
                    fOut.write(f"{specObs.ID}  " +  '   '.join( f"{l:.3f}" for l in labelsFit  ) + f"  {bestFitChi2:.3f}   {snr:.1f}"  + '\n')
            #profiler.disable()
            #stats = pstats.Stats(profiler).sort_stats('cumulative')
            #stats.print_stats()
#profiler.disable()
#with open('./log_profiler.txt', 'w') as stream:
#    stats = pstats.Stats(profiler, stream = stream).sort_stats('cumulative')
#    stats.print_stats()
