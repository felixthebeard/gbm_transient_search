
#!/usr/bin/env python3

##################################################################
# Generic script to fit the physical background model for GBM
#
# Optional command line arguments:
# -c --config_file
# -cplot --config_file_plot
# -dates --dates
# -dets --detectors
# -e --echans
# -trig --trigger
#
# Run with mpi:
# mpiexec -n <nr_cores> python fit_background.py \
#                       -c <config_path> \
#                       -cplot <config_plot_path> \
#                       -dates <date1> <date2> \
#                       -dets <det1> <det2> \
#                       -e <echan1> <echan2> \
#
# Example using the default config file:
# mpiexec -n 4 python fit_background.py -dates 190417 -dets n1 -e 2
##################################################################

from datetime import datetime

start = datetime.now()

import matplotlib

matplotlib.use("Agg")

import os
import yaml
import argparse

from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator, TrigdatBackgroundModelGenerator
from gbmbkgpy.minimizer.multinest_minimizer import MultiNestFit
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.io.export import DataExporter

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


############## Argparse for parsing bash arguments ################
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-dates", "--dates", type=str, nargs="+", help="Date string")
parser.add_argument("-dtype", "--data_type", type=str, help="Name detector", required=True)
parser.add_argument("-dets", "--detectors", type=str, nargs="+", help="Name detector", required=True)
parser.add_argument("-e", "--echans", type=int, nargs="+", help="Echan number", required=True)
parser.add_argument("-trig", "--trigger", type=str, help="Name of trigger")
parser.add_argument("-out", "--output_dir", type=str, help="Path to the output directory to continue a stopped fit")

args = parser.parse_args()

config_fit = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config_fit.yml"
)

config_plot = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config_result_plot.yml"
)

# Load the config.yml
with open(config_fit) as f:
    config = yaml.load(f)

############# Overwrite config with BASH arguments ################
config["general"]["dates"] = args.dates

config["general"]["data_type"] = args.data_type

config["general"]["detectors"] = args.detectors

config["general"]["echans"] = args.echans

if args.trigger is not None:
    config["general"]["trigger"] = args.trigger

############## Generate the GBM-background-model ##################
start_precalc = datetime.now()

if config["general"]["data_type"] in ["ctime", "cspec"]:

    model_generator = BackgroundModelGenerator()

    model_generator.from_config_dict(config)

elif config["general"]["data_type"] == "trigdat":

    model_generator = TrigdatBackgroundModelGenerator()

    model_generator.from_config_dict(config)

    model_generator.likelihood.set_grb_mask(
        f"{model_generator.data.trigtime - 15}-{model_generator.data.trigtime + 100}"
    )
else:
    raise KeyError(f"Invalid data_type used: {config['general']['data_type']}")

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

################## Plotting ########################################
start_plotting = datetime.now()

if config["plot"].get("result_plot", True):

    if rank == 0:
        print("Start plotting")

        plot_generator = ResultPlotGenerator.from_result_file(
            config_file=config_plot,
            result_data_file=os.path.join(output_dir, result_file_name),
        )

        # If we fit the background to trigdat data, then we will highlight the active
        # time, that we excluded from the fit, in the plot
        if config["general"]["data_type"] == "trigdat":
            plot_generator.add_occ_region(
                occ_name="Active Time",
                time_start=model_generator.data.trigtime - 15,
                time_stop=model_generator.data.trigtime + 150,
                time_format="MET",
                color="red",
                alpha=0.1,
            )

        plot_generator.create_plots(output_dir=output_dir)

if config["plot"].get("corner_plot", True):
    # Create corner plot
    minimizer.create_corner_plot()

comm.barrier()

stop_plotting = datetime.now()

################## Save Config ########################################
if rank == 0:

    # Save used config file to output directory
    with open(os.path.join(output_dir + "used_config.yml"), "w") as file:
        documents = yaml.dump(config, file)

if rank == 0:
    # Print the duration of the script
    print("The precalculations took: {}".format(stop_precalc - start_precalc))
    print("The fit took: {}".format(stop_fit - start_fit))
    print("The result export took: {}".format(stop_export - start_export))
    print("The plotting took: {}".format(stop_plotting - start_plotting))
    print("Whole calculation took: {}".format(datetime.now() - start))
