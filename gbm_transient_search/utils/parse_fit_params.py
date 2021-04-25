import re


def parse_bkg_fit_params(bkg_fit_params, dets, echans):

    params = {}
    for d in dets:
        params[d] = {}
        for e in echans:
            params[d][e] = []

    for (param_name, param) in param_info.items():

        normalization = re.search("^norm", param_name) is not None

        det = None
        echan = None

        if re.search("^norm_cgb", param_name) is not None:
            source_name = "cgb"
            echan = "all"
            det = "all"

        elif re.search("^norm_earth_albedo", param_name) is not None:
            source_name = "earth_albedo"
            echan = "all"
            det = "all"

        elif re.search("(?<=norm_)(.*)(?=_pl)", param_name) is not None:
            source_name = re.search("(?<=norm_)(.*)(?=_pl)", param_name)[0]
            echan = "all"
            det = "all"

        elif re.search("^norm_magnetic", param_name) is not None:
            source_name = "cosmic_rays"

        elif re.search("^norm_constant", param_name) is not None:
            source_name = "constant"

        if det is None:
            assert echan is None
            det = re.search("(?<=[01234567]_)(n[0123456789ab])$", param_name)[0]
            echan = re.search("(?<=echan-)(.*)(?=_)", param_name)[0]

        param["param_name"] = param_name
        param["source_name"] = source_name
        param["normalization"] = normalization
        param["det_name"] = det
        param["echan"] = echan

        if det == "all":
            assert echan == "all"
            for d in dets:
                for e in echans:
                    param["det_name"] = d
                    param["echan"] = e
                    params[d][e].append(param)
        else:
            params[det][echan].append(param)

    return params
