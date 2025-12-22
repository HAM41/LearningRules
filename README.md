# Inferring learning rules from decision-making data

Inferring learning rules from sensory decision making data, focused on two-alternate force choice tasks. See our associated preprint, [Inferring learning rules during de novo task learning](https://www.biorxiv.org/content/10.1101/2025.09.29.679295v1), for broad background and our modeling approach.

All inference is done at a single animal level. To this end, most methods and fitting procedures revolve around the `Trajectory` data class (in `ibl.py`), with attributes
- `X`: Regressors per trial, of shape `(T, D)`
- `Y`: Animal's choice per trial, of shape `(T,)`
- `R`: Animal's, or model's, reward per trial, of shape `(T,)`
- `day_flags`: Boolean flag indicating whether trial t is the beginning of a new session/day, boolean of shape `(T,)`

To use this code on another dataset, format the data in this `Trajectory` tuple and you should be able to get started on model fitting right away. 

## Installation

**Option 1**: (preferred) Create a virtual environment using anaconda and activate it
```bash
conda env create -f environment.yml
conda activate learning
```
This should take below 30 minutes on most modern Linux systems and clusters.

**Option 2**: For plain venv/pip, use `requirements.txt`. 

## Execution

### Model fitting

To train a model, run `fit.py`. It accepts primarily the following arguments:
- `N`: number of particles for SMC estimation of the marginal log-lik
- `lab`: IBL lab to load from
- `subject-id`: id within lab
- `model-class`: See file for options.

For example: to train a vectorized GLM policy model with policy-gradient learning dynamics, and output to `output.log`, run the following from your command line.
```
python3 fit.py --N=5000 --model-class='RVBF' --vector-alpha > output.log 2>&1
```
The run time varies with recording length, but be within hours for typical recordings of >10,000 trials on academic clusters. To test execution on your machine, please see `demos/fit_demo.py`, which is a simpler fitting script on a randomly generated dataset.  

This implementation currently *does not* save parameters per say, since the parameter count is always small we've been working with output files. You have to keep track of the logging file where this information is printed, and have it retrieved by other files. 

### Model postprocessing


Most postprocessing is done with `postprocessing/compute_posterior.py`. It has he following main methods:
- `--evalute`: compute log-liks on full trajectory and test trials
- `--posterior`: generate posterior trajectories
- `--prior`: generate model prior predictive trajectories
- `--load-from-file`: logging file containing the printed parameters

To evaluate a specific fitted model, you must provide the same arguments that were called for `fit.py`, as well as a pointer to the output file containing the logged parameters. For instance, to evaluate the model we trained above, given outputs (incl. parameters) in the file `output.log`, run
```
python3 postprocessing/compute_posterior.py \
   --N=5000 --model-class='RVBF' --vector-alpha \
   --load-from-file='output.log' --evaluate
```

The run time varies with recording length, but be within minutes for typical recordings of >10,000 trials on academic clusters. 