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
