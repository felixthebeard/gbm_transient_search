# GBM Transient Search Pipeline

A pipeline to automatically search for untriggered long duration transients in the data of Fermi-GBM.

This uses the physical background model for the GBM instrument.

If you are interested in the science behind the background model, checkout our publication: [Biltzinger et al. 2017](https://www.aanda.org/articles/aa/pdf/forth/aa37347-19.pdf)


### Setup

- Create a virtual environment

- Install threeML

- Install gbm_drm_gen https://github.com/grburgess/gbm_drm_gen (Needs the branch https://github.com/fkunzweiler/gbm_drm_gen/tree/ctime_drm_gen)

- Install gbmbkgpy https://github.com/mpe-heg/GBM-background-model

- Install package gbm_transient_search
  ```
  git clone https://github.com/fkunzweiler/gbm_transient_search
  cd gbm_transient_search
  pip install .
  ```
   
- Define secrets env variables

``` bash
export GBMBKGPY_FORCE_ABORT_ON_EXCEPTION=True
export GBM_TRANSIENT_SEARCH_AUTH_TOKEN=<token>
export GBM_TRANSIENT_SEARCH_BASE_URL=https://grb.mpe.mpg.de
export SLACK_BOT_TOKEN=<token>
```
  
### Run
1. Open master connections to cluster
  ```
  ./scripts/create_master.sh raven 3
  ```

2. Run Search
   
- Search last day:
   ```
   ./bin/transient_search_cron.sh
   ```
- Search last 5 days:
  ```
  ./bin/transient_search_cron_range.sh
  ```
  
3. For continuous search define cronjob:

``` bash
env >> ~/.cron_vars

crontab -e
```

```
0 2 * * * . /home/fkunzwei/.cron_vars; /home/fkunzwei/data1/sw/gbm_transient_search/bin/transient_search_cron.sh >> /home/fkunzwei/log/transient_search.log 2>&1
0 6 * * * . /home/fkunzwei/.cron_vars; /home/fkunzwei/data1/sw/gbm_transient_search/bin/transient_search_cron_range.sh >> /home/fkunzwei/log/transient_search_range.log 2>&1
```

