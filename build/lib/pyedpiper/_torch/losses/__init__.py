import importlib as _imp
import os as _os


# Automatically import any Python files in the losses directory
for file in _os.listdir(_os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        _imp.import_module(f"{__name__}." + file[:file.find('.py')])
