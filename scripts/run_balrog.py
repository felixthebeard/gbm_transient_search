import sys
import warnings
from mpi4py import MPI
import os
from gbm_bkg_pipe.balrog.utils.fit import BalrogFit
from gbmbkgpy.utils import global_exept_hook

global_exept_hook.add_hook()

warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

trigger_name = sys.argv[1]
trigger_info_file = sys.argv[2]

# get fit object
multinest_fit = BalrogFit(trigger_name, trigger_info_file)

multinest_fit.fit()
multinest_fit.save_fit_result()
multinest_fit.create_spectrum_plot()
multinest_fit.unlink_temp_chains_dir()

success_file = os.path.join(
    os.path.dirname(trigger_info_file),
    f"{trigger_name}_balrog.success"
)

os.system(f"touch {success_file}")

