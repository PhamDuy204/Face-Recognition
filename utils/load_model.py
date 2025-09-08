import importlib
import os
import sys
import importlib.util
sys.path.append(os.path.dirname(__file__))
def load_model(model_name='vit'):
    spec = importlib.util.spec_from_file_location(f"{model_name}", f'models/{model_name}/{model_name}.py')
    if spec:
        my_module = importlib.util.module_from_spec(spec)
        if my_module:
            spec.loader.exec_module(my_module)
        return my_module.FaceNet