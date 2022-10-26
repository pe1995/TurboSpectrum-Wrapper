from sys import argv, exit
import numpy as np
from multiprocessing import Pool
import glob
from observations import readSpectrumTSwrapper
import h5py
import cProfile
import pstats
from multiprocessing import Pool

def gather_data( arg ):
    specList, wvl, labels, quite = arg
    # spectraList, wave_new, Rnew, quite, limits = arg
    spectraIncl = []
    # get the labels
    totSize = len(specList)
    labels = { k:np.full( totSize, np.nan ) for k in labels}
    flxl = np.full( ( totSize, len(wvl) ), np.nan )
    names = []
    ## TODO:
    comments = []

    for i, specFile in enumerate(specList):
        if not quite:
            print(f"{1e2 * i/len(specList):.1f} %")
        spec = readSpectrumTSwrapper(specFile)
        spec.id = specFile.split('/')[-1]
        names.append(spec.id)

        flxl[i] = spec.flux
        for k in labels:
            labels[k][i] = spec.__dict__[k]

    return (flxl, labels, wvl, names)

if __name__ == '__main__':

    path = argv[1]
    if len(argv) > 2:
        ncpu = int(argv[2])
    else:
        ncpu = 1
    specList = glob.glob(path)
    size = len(specList)
    print(f"found {len(specList):.0f} spectra")

    # profiler = cProfile.Profile()
    # profiler.enable()
    spec = readSpectrumTSwrapper(specList[0])
    wvl = spec.lam
    labels = spec.labels

    args = [ [specList[i::ncpu], wvl, labels, True] for i in range(ncpu)]
    # unmute one sub-process
    args[0][-1] = False
    with Pool(processes=ncpu) as pool:
        out = pool.map(gather_data, args )

    flxl = np.vstack( list(out[i][0] for i in range(len(out))) )
    labels = { k : np.hstack( list(out[i][1][k] for i in range(len(out))) ) for k in out[0][1]}
    names = []
    for i in range(len(out)):
        names.extend( out[i][3] )


    with h5py.File('./test.h5', 'w') as hf:
        hf.create_dataset( 'fluxes', data=flxl, shape=np.shape(flxl), dtype='float64')
        for k in labels:
            hf.create_dataset( f"{k}", data=labels[k],  dtype='float64')
        hf.create_dataset( 'wave', data=out[0][2], dtype='float64')
#        hf.create_dataset( 'ID', data=names, dtype=str)
       # need to convert to bytes?

    # with h5py.File('./test.h5', 'r') as hf:
    #     print( list(hf.keys()) )
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats()



    exit()
