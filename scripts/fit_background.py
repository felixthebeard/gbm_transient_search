#!/usr/bin/env python3
from datetime import datetime

time_start = datetime.now()

import os
import time
import json
import numpy as np
import yaml
import arviz

from gbmbkgpy.io.export import PHAWriter, StanDataExporter
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.utils.stan import StanDataConstructor, StanModelConstructor

from gbmbkgpy.io.package_data import get_path_of_external_data_dir

from cmdstanpy import cmdstan_path, CmdStanModel

############## Argparse for parsing bash arguments ################
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-c",
    "--config_file",
    type=str,
    help="Name of the config file located in gbm_data/fits/",
)
parser.add_argument("-dates", "--dates", type=str, nargs="+", help="Date string")
parser.add_argument("-outdir", "--output_dir", type=str, help="Output directory")
parser.add_argument(
    "--export_whole_day",
    action="store_true",
    help="Export the entire day including the saa regions",
)

args = parser.parse_args()


# Load the config.yml
with open(args.config_file) as f:
    config = yaml.load(f)

############# Overwrite config with BASH arguments ################
if args.dates is not None:
    config["general"]["dates"] = args.dates

if args.output_dir is not None:
    output_dir = args.output_dir

else:
    output_dir = os.path.join(
        get_path_of_external_data_dir(),
        "bkg_pipe",
        args.dates[0],
        config["general"]["data_type"],
        "phys_bkg",
        f"det_{'_'.join(config['general']['detectors'])}",
        f"e{'_'.join(config['general']['echans'])}",
    )

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


start_mg = datetime.now()

model_generator = BackgroundModelGenerator()
model_generator.from_config_dict(config)

time_mg = datetime.now() - start_mg

##############################################
# Stan model caching
stan_model_const = StanModelConstructor(model_generator=model_generator)
source_count = stan_model_const.source_count()

stan_model_db = "/u/fkunzwei/scripts/bkg_pipe/stan_models/"
stan_model_dir = os.path.join(
    stan_model_db,
    f"{'free_earth' if source_count['use_free_earth'] else 'fixed_earth'}",
    f"{'free_cgb' if source_count['use_free_cgb'] else 'fixed_cgb'}",
    f"{source_count['num_free_ps']}_free_ps",
    f"{source_count['num_saa_exits']}_saa_exits",
    f"{source_count['num_cont_sources']}_cont_src",
    f"{source_count['num_fixed_global_sources']}_fix_gl_src",
)

stan_model_file = os.path.join(stan_model_dir, "background_model.stan")

if not os.path.exists(stan_model_file):

    if not os.path.exists(stan_model_dir):
        os.makedirs(stan_model_dir)

    stan_model_const.create_stan_file(stan_model_file)

##############################################


# Create Stan Model
model = CmdStanModel(stan_file=stan_model_file, cpp_options={"STAN_THREADS": "TRUE"})

n_cores_stan = 10

# StanDataConstructor
stan_data = StanDataConstructor(
    model_generator=model_generator, threads_per_chain=n_cores_stan
)

data_dict = stan_data.construct_data_dict()

start_fit = datetime.now()

# Sample
stan_fit = model.sample(
    data=data_dict,
    output_dir=os.path.join(output_dir, "stan_chains"),
    chains=4,
    seed=int(np.random.rand() * 10000),
    parallel_chains=4,
    threads_per_chain=n_cores_stan,
    iter_warmup=1200,
    iter_sampling=300,
    show_progress=True,
)

time_fit = datetime.now() - start_fit

start_export = datetime.now()

# Export fine binned data
config = model_generator.config
# Create a copy of the response precalculation
response_precalculation = model_generator._resp

# Create a copy of the geomtry precalculation
geometry_precalculation = model_generator._geom

# Create copy of config dictionary
config_export = config

config_export["general"]["min_bin_width"] = 5
config_export["mask_intervals"] = []

# Create a new model generator instance of the same type
model_generator_export = type(model_generator)()

model_generator_export.from_config_dict(
    config=config_export,
    response=response_precalculation,
    geometry=geometry_precalculation,
)
# StanDataConstructor
stan_data_export = StanDataConstructor(
    model_generator=model_generator_export, threads_per_chain=n_cores_stan
)

data_dict_export = stan_data_export.construct_data_dict()


##################################################
# Create stan export model if not already existing
stan_model_file_export = os.path.join(stan_model_dir, "background_model_export.stan")
if not os.path.exists(stan_model_file_export):

    if not os.path.exists(stan_model_dir):
        os.makedirs(stan_model_dir)

    stan_model_const.create_stan_file(stan_model_file_export, total_only=True)
##################################################


# Create Stan Model
model_export = CmdStanModel(
    stan_file=stan_model_file_export, cpp_options={"STAN_THREADS": "TRUE"}
)

model_export.compile()

export_quantities = model_export.generate_quantities(
    data=data_dict_export,
    mcmc_sample=stan_fit,
    gq_output_dir=os.path.join(output_dir, "stan_chains"),
)


stan_data_export = StanDataExporter.from_generated_quantities(
    model_generator_export,
    export_quantities,
    stan_fit=stan_fit,
    param_lookup=stan_data.param_lookup,
)

result_file_name = "fit_result_{}_{}_e{}.hdf5".format(
    config["general"]["dates"][0],
    "-".join(config["general"]["detectors"]),
    "-".join(config["general"]["echans"]),
)

stan_data_export.save_data(file_path=os.path.join(output_dir, result_file_name))

if args.export_whole_day:
    config_export["saa"]["time_after_saa"] = 100
    config_export["saa"]["time_before_saa"] = 30
    config_export["saa"]["short_time_intervals"] = True

    # Create a new model generator instance of the same type
    model_generator_export = type(model_generator)()

    model_generator_export.from_config_dict(
        config=config_export,
        response=response_precalculation,
        geometry=geometry_precalculation,
    )
    # StanDataConstructor
    stan_data_export = StanDataConstructor(
        model_generator=model_generator_export, threads_per_chain=n_cores_stan
    )

    data_dict_export = stan_data_export.construct_data_dict()

    export_quantities = model_export.generate_quantities(
        data=data_dict_export,
        mcmc_sample=stan_fit,
        gq_output_dir=os.path.join(output_dir, "stan_chains"),
    )

    stan_data_export = StanDataExporter.from_generated_quantities(
        model_generator_export,
        export_quantities,
        stan_fit=stan_fit,
        param_lookup=stan_data.param_lookup,
    )

    result_file_name = "fit_result_total_{}_{}_e{}.hdf5".format(
        config["general"]["dates"][0],
        "-".join(config["general"]["detectors"]),
        "-".join(config["general"]["echans"]),
    )

    stan_data_export.save_data(file_path=os.path.join(output_dir, result_file_name))

time_export = datetime.now() - start_export

start_arviz = datetime.now()

# Build arviz object
arviz_result = arviz.from_cmdstanpy(
    posterior=stan_fit,
    posterior_predictive="ppc",
    observed_data={"counts": data_dict["counts"]},
    constant_data={
        "dates": config["general"]["dates"],
        "time_bins": data_dict["time_bins"],
        "dets": model_generator.data.detectors,
        "echans": model_generator.data.echans,
        "global_param_names": stan_data.global_param_names,
        "cont_param_names": stan_data.cont_param_names,
        "saa_param_names": stan_data.saa_param_names,
    },
    predictions=stan_model_const.generated_quantities(),
)

arviz_file_name = "fit_result_{}_{}_e{}.nc".format(
    config["general"]["dates"][0],
    "-".join(config["general"]["detectors"]),
    "-".join(config["general"]["echans"]),
)

# Save this object
arviz_result.to_netcdf(os.path.join(output_dir, arviz_file_name))

time_arviz = datetime.now() - start_arviz

print(f"The model generation took: {time_mg}")
print(f"The stan fit took: {time_fit}")
print(f"The export took: {time_export}")
print(f"The arviz export took: {time_arviz}")
print(f"The total runtime was: {datetime.now() - time_start}")

os.system(f"touch {os.path.join(output_dir, 'success.txt')}")
