import models
import argparse

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