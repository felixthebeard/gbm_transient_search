#!/usr/bin/env python3

##################################################################
# Generic script to fit the physical background model for GBM
#
# Optional command line arguments:
# -c --setup_config
# -out --output_dir
#
# Run with mpi:
# mpiexec -n <nr_cores> python fit_background.py \
#                       -setup_config <config_path> \
#                       -out <output_path>
#
# Example using the default config file:
# mpiexec -n 4 python fit_background.py -dates 190417 -dets n1 -e 2
##################################################################

from datetime import datetime

start = datetime.now()

import os
import yaml
import argparse

from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.io.export import DataExporter

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


############## Argparse for parsing bash arguments ################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "-c", "--config_file", type=str, help="Path to the config file", required=True
)
parser.add_argument(
    "-out",
    "--output_dir",
    type=str,
    help="Path to the output directory to continue a stopped fit",
    required=True,
)

args = parser.parse_args()

# Load the config.yml
with open(args.config_file) as f:
    config = yaml.load(f)

############## Generate the GBM-background-model ##################
start_precalc = datetime.now()

model_generator = BackgroundModelGenerator()

model_generator.from_config_dict(config)

comm.barrier()

stop_precalc = datetime.now()

############### Instantiate Minimizer #############################
start_fit = datetime.now()

if config["fit"]["method"] == "multinest":
    minimizer = MultiNestFit(
        likelihood=model_generator.likelihood,
        parameters=model_generator.model.free_parameters,
    )

    # Fit with multinest and define the number of live points one wants to use
    minimizer.minimize_multinest(
        n_live_points=config["fit"]["multinest"]["num_live_points"],
        const_efficiency_mode=config["fit"]["multinest"]["constant_efficiency_mode"],
        output_dir=args.output_dir,
    )

else:

    raise KeyError("Invalid fit method")

# Minimizer Output dir
output_dir = minimizer.output_dir

comm.barrier()

stop_fit = datetime.now()

################# Data Export ######################################
start_export = datetime.now()

data_exporter = DataExporter(
    model_generator=model_generator, best_fit_values=minimizer.best_fit_values,
)

result_file_name = "fit_result.hdf5"

data_exporter.save_data(
    file_path=os.path.join(output_dir, result_file_name),
    result_dir=output_dir,
    save_ppc=config["export"]["save_ppc"],
)

stop_export = datetime.now()

if rank == 0:
    # Print the duration of the script
    print("The precalculations took: {}".format(stop_precalc - start_precalc))
    print("The fit took: {}".format(stop_fit - start_fit))
    print("The result export took: {}".format(stop_export - start_export))
    print("Whole calculation took: {}".format(datetime.now() - start))
