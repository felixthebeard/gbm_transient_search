#!/bin/bash
#/home/fkunzwei/data1/envs/bkg_pipe/bin/python -m luigi --module gbm_bkg_pipe CreateReportDate --workers 32 --date $(date +'%Y-%m-%d')
source /home/fkunzwei/data1/envs/bkg_pipe/bin/activate

/home/fkunzwei/data1/envs/bkg_pipe/bin/python -m luigi --workers 32 --module gbm_bkg_pipe CreateReportDate --date $(date --date='-1 day' +'%Y-%m-%d')
