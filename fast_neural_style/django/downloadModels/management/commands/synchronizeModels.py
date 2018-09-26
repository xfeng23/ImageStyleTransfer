from django.core.management.base import BaseCommand, CommandError
import os, sys
#lib_path = os.path.abspath(os.path.join('..', '..'))
#sys.path.append(lib_path)
from fast_neural_style.helpers import *
from datetime import datetime

MODELDIR = '../production_models'

class Command(BaseCommand):
    args = ''
    help = 'Closes the specified poll for voting'

    def handle(self, *args, **options):
        # print date and time
        
        print(str(datetime.now()))
        print("checking new models.")
        downloads, deletes = check_new_models(MODELDIR)
        print("download model: {}".format(downloads))
        print("delete model: {}".format(deletes))
        # download models
        download_models(downloads, MODELDIR)
	# delete models
        delete_models(deletes, MODELDIR) 
        print("\n")

