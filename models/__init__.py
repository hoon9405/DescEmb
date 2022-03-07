"""
1. import and define all model classes
2. run register_model() for each class, and add model class to MODEL_REGISTRY
3. when model.build_model() is executed, check if model name exists in MODEL_REGISTRY
"""
import importlib
import os
import pdb

MODEL_REGISTRY = {}
# should be: MODEL_REGISTRY = {}
# pdb.set_trace()

# build model instance
# only execute build_model() if model argument passed by user is defined
# for main model (not embed model, predict model)
def build_model(args):
    model = None
    model_type = getattr(args, "model", None)

    if model_type in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_type]

    assert model is not None, (
        f"Could not infer model type from {model_type}. "
        f"Available models: "
        + str(MODEL_REGISTRY.keys())
        + " Requested model type: "
        + model_type
    )

    return model.build_model(args)

# a function decorator that adds model name, class to MODEL_REGISTRY
def register_model(name):
    """
    New model types can be added with the :func:`register_model`
    function decorator.

    For example:

        @register_model('descemb_bert')
        class BertTextEncoder(nn.Module):
            (...)

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")

        MODEL_REGISTRY[name] = cls

        return cls

    return register_model_cls

def import_models(models_dir, namespace):
    # for each file in models/ directory
    # : __pycache__, __init__.py, codeemb.py, descemb.py, ehr_model.py, rnn.py, word2vec.py
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            # model_name: codeemb, descemb, ehr_model, rnn, word2vec 
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            # models.codeemb, models.descemb, models.ehr_model, models.rnn, models.word2vec
            importlib.import_module(namespace + "." + model_name)

# automatically import any Python files in the models/ directory
# models_dir: models/ directory
models_dir = os.path.dirname(__file__)
# 1. import and define all model classes
# 2. run register_model() for each class, and add model instance to MODEL_REGISTRY
# 3. when model.build_model is executed, check if model name exists in MODEL_REGISTRY
import_models(models_dir, "models")

# MODEL_REGISTRY dictionary should be filled with models
# each model class instance will be created
# pdb.set_trace()