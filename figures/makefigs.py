import numpy as np
import matplotlib.pyplot as plt
import os

def save_foolratio_fig(X, fname, fool_dict, legend=True):
    epsilon = X[:,fool_dict['epsilon']]
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(epsilon, X[:, fool_dict['DeepFool']] * 100., '-ro', markersize=8)
    ax.plot(epsilon, X[:, fool_dict['Alg1']] * 100., '-bx', markersize=10)
    ax.plot(epsilon, X[:, fool_dict['Alg2']] * 100., '-g*', markersize=10)
    ax.plot(epsilon, X[:,fool_dict['FGS']]*100., '-ms')
    ax.plot(epsilon, X[:, fool_dict['rand']] * 100., '-^k')

    plt.xlabel('epsilon', fontsize=16)
    plt.ylabel('Fooling Ratio (in %)', fontsize=16)
    ax.grid()
    plt.ylim((-5,105))
    if legend==True:
        ax.legend(['DeepFool', 'Alg 2', 'Alg 1', 'FastGrad', 'random'], fontsize=16)
    ax.tick_params(axis='both', labelsize='large')
    plt.savefig( fname, format='jpg', dpi=500 )


fool_dict = {'epsilon':0, 'FGS':1, 'Alg1':2, 'Alg2':3, 'rand':4, 'DeepFool':5}
fname = 'figslist.txt'
os.system('ls ../results/fool_* > '+ fname)

with open(fname) as f:
	lines = f.readlines()
	for ii in range(0, len(lines)):
		if ii == 2:
			legend = True
		else:
			legend = False

		line = lines[ii].split('\n')[0]
		X = np.loadtxt(line, delimiter=";")
		save_foolratio_fig(X, 'fig'+ line.split('summary')[1].split('.')[0]+'.jpg', 
			fool_dict, legend=legend)
