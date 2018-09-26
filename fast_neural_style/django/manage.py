#!/usr/bin/env python
import os
import sys
import torch 

#lib_path = os.path.abspath(os.path.join('..', '..'))
#sys.path.append(lib_path)

from fast_neural_style.config import *
from fast_neural_style.django.StyleTransferEndpoint import backendGlobal

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "StyleTransferEndpoint.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        # The above import may fail for some other reason. Ensure that the
        # issue is really that Django is missing to avoid masking other
        # exceptions on Python 2.
        try:
            import django
        except ImportError:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            )
        raise
    backendGlobal.init_pytorch_model()
    execute_from_command_line(sys.argv)
