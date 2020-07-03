from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.bkg_fit_handler import GBMBackgroundModelFit
from gbm_bkg_pipe.trigger_search import TriggerSearch
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
