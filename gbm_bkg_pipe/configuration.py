from configya import YAMLConfig
import copy

# setup_all_sources = dict(
#     use_saa=False,
#     use_constant=True,
#     use_cr=True,
#     use_earth=True,
#     use_cgb=True,
#     fix_earth=True,
#     fix_cgb=True,
#     use_sun=False,
#     ps_list=dict(
#         auto_swift=dict(update_catalog=False, flux_limit=0.1, exclude=["Crab"]),
#         CRAB=dict(
#             fixed=True, spectrum=dict(pl=dict(spectrum_type="pl", powerlaw_index=2,))
#         ),
#     ),
#     cr_approximation="BGO",
#     use_eff_area_correction=False,
# )

# # run_detectors = [
# #     ["n0"],
# #     ["n1"],
# #     ["n2"],
# #     ["n3"],
# #     ["n4"],
# #     ["n5"],
# #     ["n6"],
# #     ["n7"],
# #     ["n8"],
# #     ["n9"],
# #     ["na"],
# #     ["nb"],
# #     ["b0"],
# #     ["b1"],
# # ]

# # run_echans = [
# #     ["0"],
# #     ["1"],
# #     ["2"],
# #     ["3"],
# #     ["4"],
# #     ["5"],
# #     ["6"],
# #     ["7"],
# # ]

run_detectors = [
    ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"],
]
run_echans = [
    ["0", "1", "2", "3", "4"],
    ["5", "6", "7"]
]
# bkg_source_setup = {}
# for echans in run_echans:
#     bkg_source_setup["_".join(echans)] = copy.deepcopy(setup_all_sources)

# bkg_source_setup["0_1_2_3_4"].update(dict(use_cr=True))
# bkg_source_setup["5_6_7"].update(dict(ps_list=[]))

# bkg_source_setup["0"].update(dict(use_cr=False))
# bkg_source_setup["1"].update(dict(use_cr=False))
# bkg_source_setup["4"].update(dict(ps_list=[]))
# bkg_source_setup["5"].update(dict(ps_list=[]))
# bkg_source_setup["6"].update(dict(ps_list=[]))
# bkg_source_setup["7"].update(dict(use_cgb=False, ps_list=[]))

structure = {}

structure["luigi"] = dict(n_workers=4)
structure["phys_bkg"] = dict(
    stan=dict(n_cores=4),
    multinest=dict(n_cores=4, path_to_python="python"),
    timeout=2 * 60 * 60,  # 1 hour
    #bkg_source_setup=bkg_source_setup,
)

structure["balrog"] = dict(
    multinest=dict(n_cores=4, path_to_python="python"), timeout=1 * 60 * 60  # 1 hour
)

structure["data"] = dict(data_type="ctime", detectors=run_detectors, echans=run_echans)


class GBMBkgPipeConfig(YAMLConfig):
    def __init__(self):

        super(GBMBkgPipeConfig, self).__init__(
            structure=structure,
            config_path="~/.gbm_bkg_pipe",
            config_name="gbm_bkg_pipe_config.yml",
        )


gbm_bkg_pipe_config = GBMBkgPipeConfig()
