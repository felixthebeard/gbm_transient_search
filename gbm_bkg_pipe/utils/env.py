import os

from gbm_bkg_pipe.exceptions.custom_exceptions import ImproperlyConfigured


def get_env_value(env_variable):
    try:
        return os.environ[env_variable]
    except KeyError:
        error_msg = "Set the {} environment variable".format(env_variable)
        raise ImproperlyConfigured(error_msg)
