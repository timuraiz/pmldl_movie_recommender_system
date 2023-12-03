import os
path_to_models = os.path.abspath(os.path.dirname(__file__))
root_path = path_to_models.replace('/models', '')
print(root_path)
