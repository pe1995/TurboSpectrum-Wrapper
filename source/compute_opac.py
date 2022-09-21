# external
from sys import argv
import numpy as np
from multiprocessing import Pool
# local
from configure_setup import setup
from run_ts import compute_babsma, compute_bsyn
from atmos_package import model_atmosphere


def runTSforOpac(args):
    setup, indices = args
    for i, atmFile in enumerate(setup.atmos_list[indices]):
        " Read model atmosphere and write in TS appropriate format if not already"
        print(atmFile)
        atmos = model_atmosphere()
        atmos.read(atmFile, setup.atmos_format)
        if setup.atmos_format.strip() == 'marcs':
            setup.ts_input['MARCS-FILE'] = '.true.'
            atmos.path = atmFile
        else:
            atmos.path = f"{setup.cwd}/{atmos.ID}.dat"
            write_atmos_m1d4TS(atmos,  atmos.path)

        " Where to write opacities? "
        modelOpacFile = f"{setup.cwd}_{atmos.id}_{setup.jobID}"
        # TODO: add specific keywords to ts_input and update TS to write requested opacities
        " Run TS babsma script "
        compute_babsma(setup.ts_input, atmos, modelOpacFile, quite=setup.debug)

        " Run TS bsyn script for line opacities"
        # QUESTION: is that gonna do the trick?
        # TODO: check input to TS regards to flags
        setup.ts_input['MULTIDAMP'] = '.true.'
        elementalAbundances = [] # this will assume TS internal solar mixture for now
        specResultFile = './ignoreSpectrum.txt'
        compute_bsyn(
                    setup.ts_input, elementalAbundances, atmos, modelOpacFile,
                    specResultFile, nlteInfoFile = None, quite = True
                    )

if __name__ == '__main__':
    if len(argv) > 2:
        conf_file = argv[1]
    else:
        print("Usage: $ pytnon generate_random_grid.py ./configFile.txt jobName")
        exit()

    setup = setup(file = conf_file, mode = 'MAprovided')
    setup.jobID = argv[2]

    if 'ncpu' not in setup.__dict__:
        setup.ncpu = 1

    ind = np.arange(len(setup.atmos_list))
    args = [ [setup, ind[i::setup.ncpu]] for i in range(setup.ncpu)]
    with Pool(processes=setup.ncpu) as pool:
        pool.map(runTSforOpac, args )
