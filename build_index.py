import hnswlib
import numpy as np
import h5py
import time
from progressbar import *
import pickle


if __name__ == "__main__":
	dim = 25
	f = h5py.File('glove-25-angular.hdf5','r')

	test = np.array(f.get('test'))
	train = np.array(f.get('train'))
	neighbors = np.array(f.get('neighbors'))
	distances = np.array(f.get('distances'))

	p = hnswlib.Index(space='l2', dim=dim)

	num_elements = train.shape[0]

	p.init_index(max_elements=num_elements, ef_construction=300, M=16)

	p.set_ef(10)

	p.set_num_threads(4)

	p.add_items(train)

	# Serializing and deleting the index:
	index_path='train_index.bin'
	print("Saving index to '%s'" % index_path)
	p.save_index(index_path)
