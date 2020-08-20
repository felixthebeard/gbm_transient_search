#!/usr/bin/env python3

import os

import tempfile
from gbmbkgpy.utils.select_pointsources import build_swift_pointsource_database


with tempfile.TemporaryDirectory() as tmpdirname:
    build_swift_pointsource_database(tmpdirname, multiprocessing=True, force=True, orbit_resolution=True)
