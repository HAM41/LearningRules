import models
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from itertools import combinations

def load_model(args: argparse.Namespace):
    if args.model_class == "GLMLearn":
        model = models.GLMLearn(learning_rule=args.learning_rule)
    elif args.model_class == "TimeVarGLMLearn":
        model = models.TimeVarGLMLearn(learning_rule=args.learning_rule, lapse=args.lapse, beta_dim=1)
    elif args.model_class == "Psytrack":
        model = models.Psytrack()
    elif args.model_class == "GLMRegLearn":
        args.learning_rule = 'regression_gradient'
        model = models.GLMRegLearn(learning_rule='regression_gradient')
    elif args.model_class == "GLMHMMLearn":
        model = models.GLMHMMLearn(learning_rule=args.learning_rule)
    elif args.model_class == "GLMInterpLearn":
        model = models.GLMInterpLearn(learning_rule="interp_gradient")
    elif args.model_class == "QLearning":
        model = models.QLearning()
    elif args.model_class == "GLMBaseLearn":
        model = models.GLMBaseLearn(time_var=args.vector_alpha)
    elif args.model_class == "DynamicGLMHMM":
        model = models.DynamicGLMHMM(K=3)
    elif args.model_class == "AC":
        model = models.AC(beta_dim=1)
        args.learning_rule = 'reinforce'
    elif args.model_class == "RVBF":
        model = models.RVBF()
    elif args.model_class == "TimeVarRVBF":
        model = models.TimeVarRVBF()
    elif args.model_class == "HRL":
        model = models.HRL()
    else:
        raise ValueError(f"Model class {args.model_class} not recognized.")
    return model

def is_ndimensional_space(points):
    # Convert the list of points to a NumPy array for easier manipulation
    points_array = np.array(points)

    # Check if the number of points is at least N+1
    if len(points) < len(points_array[0]) + 1:
        return False

    # Compute vectors between the points
    vectors = jnp.array([i[0]-i[1] for i in combinations(points_array, 2)])

    # Compute the matrix rank to check for linear independence
    rank = np.linalg.matrix_rank(vectors)

    # If the rank is equal to N, the points form an N-dimensional space
    return rank == len(points_array[0])

def make_n_dimensional(points, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    # Convert the list of points to a NumPy array for easier manipulation
    points_array = np.array(points)

    # Check if the number of points is at least N+1
    if len(points) < len(points_array[0]) + 1:
        raise ValueError("The number of points must be at least N+1")

    # If already n dimensional, just return the points
    if is_ndimensional_space(points):
        return points

    # If the points do not form an N-dimensional space, perturb the last points

    # Compute vectors between the points
    vectors = jnp.array([i[0]-i[1] for i in combinations(points_array, 2)])

    # Compute the matrix rank to check for linear independence
    rank = np.linalg.matrix_rank(vectors)

    rank_deficit = len(points_array[0]) - rank
    for j in range(rank_deficit):
        key, _ = jax.random.split(key)
        perturbation = jax.random.normal(key, shape=(len(points_array[0]),))
        points_array[-j-1] += perturbation

    return points_array