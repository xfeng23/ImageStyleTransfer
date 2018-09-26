#!/bin/sh
echo Start Django Server.
# add style transfer library into PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$PWD"
# launch distribute service
cd fast_neural_style/distribute_django
python manage.py runserver 0.0.0.0:36060 &
cd ../..
# launch backend services
cd fast_neural_style/django
# set the current device
export CUDA_VISIBLE_DEVICES=0
python manage.py runserver 0.0.0.0:30006 &
export CUDA_VISIBLE_DEVICES=1
python manage.py runserver 0.0.0.0:33100 &
export CUDA_VISIBLE_DEVICES=2
python manage.py runserver 0.0.0.0:31004 &
export CUDA_VISIBLE_DEVICES=3
python manage.py runserver 0.0.0.0:35010 &
