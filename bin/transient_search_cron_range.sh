#!/bin/bash
#/home/fkunzwei/data1/envs/bkg_pipe/bin/python -m luigi --module gbm_bkg_pipe CreateReportDate --workers 32 --date $(date +'%Y-%m-%d')
source /home/fkunzwei/data1/envs/bkg_pipe/bin/activate

/home/fkunzwei/data1/envs/bkg_pipe/bin/python -m luigi --workers 32 --module gbm_transient_search RangeDailyBase --of CreateReportDate --days-back=5 --stop $(date +'%Y-%m-%d') --reverse
