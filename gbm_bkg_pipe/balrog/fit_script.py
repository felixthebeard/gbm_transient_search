import sys
import warnings

from gbm_bkg_pipe.balrog.utils.fit import BalrogFit

warnings.simplefilter("ignore")

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size > 1:
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        using_mpi = False
except:
    using_mpi = False

trigger_name = sys.argv[1]
trigger_info_file = sys.argv[2]

# get fit object
multinest_fit = BalrogFit(trigger_name, trigger_info_file)

multinest_fit.fit()
multinest_fit.save_fit_result()
multinest_fit.create_spectrum_plot()
multinest_fit.unlink_temp_chains_dir()
