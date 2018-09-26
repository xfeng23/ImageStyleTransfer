from azure.storage.blob import BlockBlobService
from fast_neural_style.config import *

"""
Singleton Class for BlockBlobService. 

# http://code.activestate.com/recipes/52558-the-singleton-pattern-implemented-with-python/
"""
class BlobServiceSingleton():
	class __BlobServiceSingleton:
		""" Implementation of the singleton interface """
		def __init__(self):
                        #print("starting")
			self.block_blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)
		
		def get_service(self):
			return self.block_blob_service

	instance = None
	def __init__(self):
		# If instance is None, instantiate private Class
		# Else do nothing
		if BlobServiceSingleton.instance is None:
			BlobServiceSingleton.instance = BlobServiceSingleton.__BlobServiceSingleton()

	def __getattr__(self, attr):
		""" Delegate access to __BlobServiceSingleton """
		return getattr(self.instance, attr)

	def __setattr__(self, attr, value):
		""" Delegate access to __BlobServiceSingleton """
		return setattr(self.instance, attr, value)

#BlobService = BlobServiceSingleton()
