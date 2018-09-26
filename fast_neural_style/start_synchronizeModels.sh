#!/bin/sh
export PYTHONPATH="${PYTHONPATH}:/app"
cd /app/fast_neural_style/django
/home/user/miniconda/envs/pytorch-py36/bin/python manage.py synchronizeModels >> /app/fast_neural_style/synchronizeModels.log 2>&1
