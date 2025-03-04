import numpy as np 
from scipy.spatial import distance

TCN1D_train_p = {'batch_size': 2,
				'no_wells':20, 
				'epochs':1000, 
				'data_flag':'M2',

				'model_name': 'VishalNet',
				'lr':0.0001,
				'get_F':0,
				'F':'WE_PreSDM',
				}

TCN1D_test_p = {'no_wells':20,
				'data_flag':'M2',
				'model_name': 'VishalNet',
				}

phy1D_train_p = {'batch_size': 2,
				'no_wells':12, 
				'epochs':50,
				'data_flag': 'M2',
				'model_name': 'CNN',
				'lr':0.0001,
				}