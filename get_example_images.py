"""
Description...

File: view_graph.py
Author: Emilio Balda <emilio.balda@ti.rwth-aachen.de>
Organization:  RWTH Aachen University - Institute for Theoretical Information Technology
"""

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import argparse
from main import pre_process_data, collect_correctly_predicted_images, get_adversarial_noise, predict_CNN
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    # ********************* DEFAULT INPUT VARIABLES (edit if necesary) *************************
    model2load = 'nin'
    models_dir = 'pretrainedmodels/'
    out_dir = 'examples/'
    data_dir = 'datasets/'
    n_images = 20
    eps = 0.05
    method2use = 'Alg1'
    # ********************* ******************************************* *************************

    parser = argparse.ArgumentParser(description="Creates tensorboard visualization files for ")
    parser.add_argument("--model2load", type=str, default=model2load,
                        help="model to be loaded: either of these --> fcnn, lenet, nin, densenet. Default value = " + model2load)
    parser.add_argument("--method2use", type=str, default=method2use,
                        help="method to be used: either of these --> Alg1, Alg2, FGS. Default value = " + method2use)
    parser.add_argument("--models-dir", type=str, default=models_dir,
                        help="Path to the directory containing the pre-trained model(s). Default value = " + models_dir)
    parser.add_argument("--out-dir", type=str, default=out_dir,
                        help="Path to the directory where the output images files will be stored. Default value = " + out_dir)
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Path to the directory containing the dataset(s). Default value = " + data_dir)
    parser.add_argument("--n-images", type=int, default=n_images,
                        help="Number of images of the dataset to be fooled. Default value = " + str(n_images))
    parser.add_argument("--epsilon", type=float, default=eps,
                        help="Number of images of the dataset to be fooled. Default value = " + str(eps))
    return parser.parse_args()

def get_all_model_variables(args):
    '''
    Description...
    '''
    if (args.model2load == 'fcnn'):
        modelvarnames = {
        'model2load':args.model2load,
        'method': args.method2use,
        'models_dir':args.models_dir,
        'out_dir':args.out_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'epsilon':args.epsilon,
        'graph_directory':'savedmodel_fcnn_mnist/',
        'graph_file':'fcnn.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'pkeep':None
        }
    elif (args.model2load == 'lenet'):
        modelvarnames = {
        'model2load':args.model2load,
        'method': args.method2use,
        'models_dir':args.models_dir,
        'out_dir':args.out_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'epsilon':args.epsilon,
        'graph_directory':'savedmodel_lenet_mnist/',
        'graph_file':'lenet.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'pkeep':None        
        }
    elif args.model2load == 'nin':
        modelvarnames = {
        'model2load':args.model2load,
        'method': args.method2use,
        'models_dir':args.models_dir,
        'out_dir':args.out_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'epsilon':args.epsilon,
        'graph_directory':'savedmodel_nin_cifar/',
        'graph_file':'nin.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'pkeep':'pkeep:0'
        }
    elif args.model2load == 'densenet':
        modelvarnames = {
        'model2load':args.model2load,
        'method': args.method2use,
        'models_dir':args.models_dir,
        'out_dir':args.out_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'epsilon':args.epsilon,
        'graph_directory':'savedmodel_densenet_cifar/',
        'graph_file':'graph-0301-145141.meta',
        'input':'input:0',
        'logits':'InferenceTower/linear/output:0',
        'pkeep':None
        }
    else:
        print('Error:  Select a valid model (fcnn, lenet, nin, densenet)')
        modelvarnames = None

    return modelvarnames

def main():
    args = get_arguments()

    allvars = get_all_model_variables(args)
    # Load Test Dataset
    if (allvars['model2load'] == 'fcnn') or (allvars['model2load'] == 'lenet'):
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets(allvars['data_dir'], one_hot=True)
        X = mnist.test.images
        y = mnist.test.labels
        labels_dict = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

        # Free Memory
        mnist = None

    if (allvars['model2load'] == 'nin') or (allvars['model2load'] == 'densenet'):
        import cifar10
        cifar10.data_path = allvars['data_dir']
        cifar10.maybe_download_and_extract()
        X, _, y = cifar10.load_test_data()
        labels_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


        # Free Memory
        cifar = None

    X, y = pre_process_data(X, y, allvars['model2load'])
    X, y = collect_correctly_predicted_images(X, y, allvars)

    eps_rescale = np.max(np.abs( np.max(X.flatten()) -  np.min(X.flatten()) ))
    
    N = get_adversarial_noise(X, y, allvars['epsilon']*eps_rescale, allvars, method=allvars['method'])
    y_adv, _ = predict_CNN(X + N, allvars)

    import scipy
    # scipy.misc.imsave('outfile.jpg', image_array*255.)
    t=0
    eps = allvars['epsilon']
    for i in range(X.shape[0]):

        Ximage = (X/eps_rescale + np.min(X.flatten()))
        Nimage = (N / eps_rescale / eps / 2 + 0.5 )
        Xadv = (1-2*eps)*Ximage + eps + 2*eps*(Nimage -0.5)
        if y_adv[i] != y[i]:
            scipy.misc.imsave(allvars['out_dir']+str(int(t))+'_Original_'+labels_dict[y[i]]+'.eps',
                              np.squeeze(Ximage[i,:,:,:])*255.)
            scipy.misc.imsave(allvars['out_dir']+str(int(t))+'_Noise_'+labels_dict[y[i]]+'.eps',
                              np.squeeze(Nimage[i,:,:,:])*255.)
            scipy.misc.imsave(allvars['out_dir']+str(int(t))+'_Adversarial_'+labels_dict[y_adv[i]]+'.eps',
                              np.squeeze(Xadv[i,:,:,:])*255.)

            t = t+1
            print(t)
        if t>=allvars['n_images']:
            break

if __name__ == '__main__':
    main()