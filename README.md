# Inferring learning rules from decision-making data

Inferring learning rules underlying sensory decision making in two-alternate force choice tasks. 

## Installation

**Option 1**: (preferred) Create a virtual environment using anaconda and activate it
```bash
conda env create -f environment.yml
conda activate learning
```

**Option 2**: For plain venv/pip, use `requirements.txt`. 

## Execution

### Model fitting

To train a model, run `fit_ibl.py`. It accepts primarily the following arguments:
- `N`: number of particles for SMC estimation of the marginal log-lik
- `lab`: IBL lab to load from
- `subject-id`: id within lab
- `model-class`: See file for options.

For example, to train a vectorized GLM policy model with policy-gradient learning dynamics, and output to `output.log`
```
python3 fit_ibl.py --N=5000 --model-class='RVBF' --vector-alpha > output.log 2>&1
```

This implementation currently *does not* save parameters. You either have to implement the saving yourself, or simply copy paste from output files.
### Model postprocessing


Most postprocessing is done with `postprocessing/compute_posterior.py`. Given the same arguments as the `fit_ibl.py`, and add the parameters to the `main()` method in `compute_posterior.py`. It has he following main methods:
- `--evalute`: compute log-liks on full trajectory and test trials
- `--posterior`: generate posterior trajectories
- `--prior`: generate model prior predictive trajectories

For instance, to evaluate the model we trained above, given parameters already included in the file, run
```
python3 postprocessing/compute_posterior.py \
   --N=5000 --model-class='RVBF' --vector-alpha \
   --evaluate
```