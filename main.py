"""
Tensorflow (version = 1.4) implementation of the following algorithms from our paper ():
    (*) Algorithms 1 and 2 	(ours)
    (*) Fast Gradient Sign 	from
    (*) DeepFool 			from
These algorithms are benchmarked using the MNIST and CIFAR10 datasets in the following pre-trained models:
    (*) FC-150-100-10				:
    (*) LeNet-5						:
    (*) Network-In-Network (NIN)	:
    (*) DenseNet					:
File: main.py
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

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    # ********************* DEFAULT INPUT VARIABLES (edit if necesary) *************************
    model2load = 'fcnn'
    models_dir = 'pretrainedmodels/'
    output_dir = 'results/'
    figs_dir = 'figures/'
    data_dir = 'datasets/'
    n_images = 128
    max_epsilon = 0.2
    # ********************* ******************************************* *************************

    parser = argparse.ArgumentParser(description="Benchmarking of Algorithms for Generation of Adversarial Examples")
    parser.add_argument("--rho-mode", help="Robustness Measure Mode: compute/store the robustness measures and exit",
                        action="store_true")
    parser.add_argument("--model2load", type=str, default=model2load,
                        help="model to be loaded: either of these --> fcnn, lenet, nin, densenet. Default value = " + model2load)
    parser.add_argument("--models-dir", type=str, default=models_dir,
                        help="Path to the directory containing the pre-trained model(s). Default value = " + models_dir)
    parser.add_argument("--output-dir", type=str, default=output_dir,
                        help="Path to the directory where the output fooling ratio(s) will be stored. Default value = " + output_dir)
    parser.add_argument("--figs-dir", type=str, default=figs_dir,
                        help="Path to the directory where the output figure(s) will be stored. Default value = " + figs_dir)
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Path to the directory containing the dataset(s). Default value = " + data_dir)
    parser.add_argument("--n-images", type=int, default=n_images,
                        help="Number of images of the dataset to be fooled. Default value = " + str(n_images))
    parser.add_argument("--max-epsilon", type=float, default=max_epsilon,
                        help="Maximum epsilon value. It must be between 0 and 1. Default value = "+ str(max_epsilon))
    return parser.parse_args()

def get_all_model_variables(args):
    '''
    Every model has its own names for the tensors in the grahp.
    After looking at the graph of each model, in tensorboard using 'ViewGraph.py',
    we create a dictionary with the tensors names of:
        (*) 'input': 	The input image of the model
        (*) 'logits': 	The output vector of the last layer of the model (before softmax is applied).
        (*)	'pkeep':	Dropout probability if applicable, None otherwise
    '''
    if (args.model2load == 'fcnn'):
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'output_dir':args.output_dir,
        'figs_dir':args.figs_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'max_epsilon':args.max_epsilon,
        'graph_directory':'savedmodel_fcnn_mnist/',
        'graph_file':'fcnn.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'pkeep':None
        }
    elif (args.model2load == 'lenet'):
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'output_dir':args.output_dir,
        'figs_dir':args.figs_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'max_epsilon':args.max_epsilon,
        'graph_directory':'savedmodel_lenet_mnist/',
        'graph_file':'lenet.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'pkeep':None
        }
    elif args.model2load == 'nin':
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'output_dir':args.output_dir,
        'figs_dir':args.figs_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'max_epsilon':args.max_epsilon,
        'graph_directory':'savedmodel_nin_cifar/',
        'graph_file':'nin.ckpt.meta',
        'input':'x:0',
        'logits':'logits:0',
        'pkeep':'pkeep:0'
        }
    elif args.model2load == 'densenet':
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'output_dir':args.output_dir,
        'figs_dir':args.figs_dir,
        'data_dir':args.data_dir,
        'n_images':args.n_images,
        'max_epsilon':args.max_epsilon,
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

def pre_process_data(X, y, model2load):
    if model2load == 'fcnn' or model2load == 'lenet':
        X = np.reshape(X, [X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])),1])
        X = np.pad(X, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    if model2load == 'nin':
        mean = np.array([125.307, 122.95, 113.865]) / 255.
        std = np.array([62.9932, 62.0887, 66.7048]) / 255.
        for i in range(3):
            X[:,:,:,i] = (X[:,:,:,i] - mean[i]) / std[i]
    if model2load == 'densenet':
        mean = np.array([125.307, 122.95, 113.865]) / 255.
        for i in range(3):
            X[:, :, :, i] = (X[:, :, :, i] - mean[i])
        X = 255.*X
    X, y = shuffle(X, y)
    return X, y

def predict_CNN(X, modelvarnames):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(modelvarnames['models_dir'] +\
                                           modelvarnames['graph_directory'] +\
                                           modelvarnames['graph_file'])
        saver.restore(sess, tf.train.latest_checkpoint(modelvarnames['models_dir'] +\
                                                       modelvarnames['graph_directory']))

        graph = tf.get_default_graph()

        Xinput = graph.get_tensor_by_name(modelvarnames['input'])
        Ylogits = graph.get_tensor_by_name(modelvarnames['logits'])
        if modelvarnames['pkeep'] == None:
            pkeep = tf.placeholder(tf.int32, (None))  # Not used
        else:
            pkeep = graph.get_tensor_by_name(modelvarnames['pkeep'])

        Ysoftmax = tf.nn.softmax(Ylogits)
        predicted_labels = tf.argmax(Ylogits, 1)
        confidence = tf.reduce_max(Ysoftmax, 1)

        Yout, yconf = sess.run([predicted_labels, confidence], feed_dict={Xinput: X, pkeep:1.0})
    return Yout, yconf

def collect_correctly_predicted_images(X, y, modelvarnames):
    Ninputs = int(1.5*modelvarnames['n_images'])
    X = X[:Ninputs,:,:,:]
    y = y[:Ninputs]
    y_pred, y_conf  = predict_CNN(X[:Ninputs,:,:,:], modelvarnames)
    print('test error = ' + str(100.*np.mean(y_pred != np.argmax(y[:Ninputs],1))) + '%')
    X = X[y_pred ==np.argmax(y[:Ninputs],1),:,:,:]
    y = y[y_pred ==np.argmax(y[:Ninputs],1)]
    X = X[:modelvarnames['n_images'],:,:,:]
    y = np.argmax(y[:modelvarnames['n_images']], 1)
    return X, y

def FastGrad(dX, epsilon):
    return -epsilon * np.sign(dX)

def get_adversarial_noise(X, ypred, epsilon, modelvarnames,
                          method='FGS', iterations=1):
    if method=='rand':
        return epsilon * np.sign(np.random.randn(X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    
    N = np.zeros(X.shape)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(modelvarnames['models_dir'] +\
                                           modelvarnames['graph_directory'] +\
                                           modelvarnames['graph_file'])
        saver.restore(sess, tf.train.latest_checkpoint(modelvarnames['models_dir'] +\
                                                       modelvarnames['graph_directory']))

        graph = tf.get_default_graph()

        Xtrue = graph.get_tensor_by_name(modelvarnames['input'])
        Ylogits = graph.get_tensor_by_name(modelvarnames['logits'])
        if modelvarnames['pkeep'] == None:
            pkeep = tf.placeholder(tf.int32, (None))  # Not used
        else:
            pkeep = graph.get_tensor_by_name(modelvarnames['pkeep'])

        Ytrue = tf.placeholder(tf.int32, (None))

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
                                                       labels=tf.one_hot(Ytrue,10))



        grad_loss = tf.gradients(loss, Xtrue)[0]
        grad = tf.stack([tf.gradients(tf.gather(Ylogits, 0, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 1, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 2, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 3, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 4, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 5, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 6, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 7, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 8, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 9, axis=1), Xtrue)[0]])


        if method == 'Alg1':
            dX_y = np.zeros(X.shape)
            N = np.zeros(X.shape)
            for i in range(X.shape[0]):
                XX = X[i,:,:,:]
                for n in range(iterations):
                    dX, fY = sess.run([grad, Ylogits], feed_dict={Xtrue: np.array([XX]), pkeep:1.0})

                    tmp_norm = float('Inf')
                    l = -1
                    for j in range(10):
                        tmp = (fY[:, ypred[i]] - fY[:, j]) - epsilon/iterations * np.abs(
                            dX[ypred[i], :, :, :, :] - dX[j, :, :, :, :]).sum()
                        if j != ypred[i] and tmp < tmp_norm:
                            tmp_norm = tmp
                            l = j

                    dX_y[i, :, :, :] = dX[ypred[i], :, :, :, :] - dX[l, :, :, :, :]

                    N[i, :, :, :] = N[i, :, :, :] + FastGrad(dX_y[i, :, :, :], epsilon/iterations)
                    XX = X[i,:,:,:] + N[i, :, :, :]
        elif method == 'FGS':
            dX_y = np.zeros(X.shape)
            for i in range(X.shape[0]):
                dX = sess.run(grad_loss, feed_dict={Xtrue: np.array([X[i, :, :, :]]), Ytrue: np.array([ypred[i]]), pkeep:1.0})
                dX_y[i, :, :, :] = dX[0, :, :, :]

            N = FastGrad(-dX_y, epsilon)
        elif method == 'Alg2':
            dX_y = np.zeros(X.shape)
            for i in range(X.shape[0]):
                dX = sess.run(grad, feed_dict={Xtrue: np.array([X[i, :, :, :]]), pkeep:1.0})

                dX_y[i, :, :, :] = dX[ypred[i], :, :, :, :]
            N = FastGrad(dX_y, epsilon)

    return N

def get_foolratio(X, y, epsilon, modelvarnames,
                  method='FGS', iterations=1):
    N = get_adversarial_noise(X, y, epsilon, modelvarnames, method=method, iterations=iterations)
    y_adv, y_conf = predict_CNN(X + N, modelvarnames)
    return np.mean(y != y_adv)


def get_deepfool_ellenorms(X, ypred, modelvarnames,
                           max_its = 50,
                           overshoot=0.02):
    # Parameters taken from --> https://github.com/LTS4/universal/blob/master/python/deepfool.py
    elle_norms = np.zeros(X.shape[0])
    fooled_vec = np.zeros(X.shape[0])
    iterations_vec = np.zeros(X.shape[0])


    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(modelvarnames['models_dir'] +\
                                           modelvarnames['graph_directory'] +\
                                           modelvarnames['graph_file'])
        saver.restore(sess, tf.train.latest_checkpoint(modelvarnames['models_dir'] +\
                                                       modelvarnames['graph_directory']))

        graph = tf.get_default_graph()

        Xtrue = graph.get_tensor_by_name(modelvarnames['input'])
        Ylogits = graph.get_tensor_by_name(modelvarnames['logits'])
        if modelvarnames['pkeep'] == None:
            pkeep = tf.placeholder(tf.int32, (None))  # Not used
        else:
            pkeep = graph.get_tensor_by_name(modelvarnames['pkeep'])
        predicted_label = tf.argmax(Ylogits, 1)

        grad = tf.stack([tf.gradients(tf.gather(Ylogits, 0, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 1, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 2, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 3, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 4, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 5, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 6, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 7, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 8, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 9, axis=1), Xtrue)[0]])

        
        for i in range(X.shape[0]):
            X0 = np.array([X[i,:,:,:]])
            N = np.zeros(X0.shape)
            XX = X0 + N
            for n in range(max_its):
                dX, fY = sess.run([grad, Ylogits], feed_dict={Xtrue: XX, pkeep: 1.0})

                tmp_norm = float('Inf')
                l=-1
                for j in range(10):
                    if j != ypred[i]:
                        tmp = np.abs(fY[:,ypred[i]] - fY[:,j])/np.abs(dX[ypred[i],:,:,:,:] - dX[j,:,:,:,:]).sum()
                        if tmp < tmp_norm:
                            tmp_norm = tmp
                            l = j

                N = N + (1 + overshoot)*tmp_norm*np.sign( dX[l,:,:,:,:]  - dX[ypred[i],:,:,:,:])
                XX = X0 + N

                ypred_i_n = sess.run(predicted_label, feed_dict={Xtrue: XX, pkeep: 1.0})
                if ypred_i_n != ypred[i]:
                    fooled_vec[i] = 1
                    break
            elle_norms[i] = np.max(np.abs(N).flatten())
            iterations_vec[i] = n

    return elle_norms, fooled_vec, iterations_vec


def get_robustness_metrics(X, ypred, elle_norms, fooled_vec, modelvarnames,
                          method='rho_1'):


    N = np.zeros(X.shape)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(modelvarnames['models_dir'] + \
                                           modelvarnames['graph_directory'] + \
                                           modelvarnames['graph_file'])
        saver.restore(sess, tf.train.latest_checkpoint(modelvarnames['models_dir'] + \
                                                       modelvarnames['graph_directory']))

        graph = tf.get_default_graph()

        Xtrue = graph.get_tensor_by_name(modelvarnames['input'])
        Ylogits = graph.get_tensor_by_name(modelvarnames['logits'])
        if modelvarnames['pkeep'] == None:
            pkeep = tf.placeholder(tf.int32, (None))  # Not used
        else:
            pkeep = graph.get_tensor_by_name(modelvarnames['pkeep'])

        Ytrue = tf.placeholder(tf.int32, (None))

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
                                                       labels=tf.one_hot(Ytrue, 10))

        grad_loss = tf.gradients(loss, Xtrue)[0]
        grad = tf.stack([tf.gradients(tf.gather(Ylogits, 0, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 1, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 2, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 3, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 4, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 5, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 6, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 7, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 8, axis=1), Xtrue)[0],
                         tf.gradients(tf.gather(Ylogits, 9, axis=1), Xtrue)[0]])



        out = None
        if method=='rho1':
            rho_1 = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                if fooled_vec[i]:
                    rho_1[i] = elle_norms[i]/np.max(np.abs(X[i, :, :, :]).flatten())
                else:
                    rho_1[i] = -1
            rho_1 = rho_1[rho_1 >0]
            out = np.mean(rho_1)
        elif method == 'rho2':
            rho_2 = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                dX, fY = sess.run([grad, Ylogits], feed_dict={Xtrue: np.array([X[i, :, :, :]]), pkeep: 1.0})

                tmp_norm = float('Inf')
                l = -1
                for j in range(10):
                    if j!= ypred[i]:
                        tmp = (fY[:, ypred[i]] - fY[:, j])/np.sum( np.abs(dX[ypred[i], :, :, :, :] - dX[j, :, :, :, :]).flatten())
                        if tmp < tmp_norm:
                            tmp_norm = tmp

                rho_2[i] = tmp_norm

            out = np.mean(rho_2)

        elif method == 'eps99':
            epsvec = np.linspace(0, np.max(elle_norms[fooled_vec == 1]), X.shape[0])
            for i in range(X.shape[0]):
                if np.mean(np.array((elle_norms<epsvec[i]) * (fooled_vec==1))) > 0.99:
                    out = epsvec[i]
                    break
        else:
            print('Non-valid robustness metric selected....')
    return out

def save_foolratio_fig(X, fname, fool_dict, legend=True):
    epsilon = X[:,fool_dict['epsilon']]
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(epsilon, X[:, fool_dict['DeepFool']] * 100., '-ro', markersize=8)
    ax.plot(epsilon, X[:, fool_dict['Alg1-10']] * 100., '--bx')
    ax.plot(epsilon, X[:, fool_dict['Alg1-5']] * 100., '-.bx')
    ax.plot(epsilon, X[:, fool_dict['Alg1']] * 100., '-bx', markersize=10)
    ax.plot(epsilon, X[:, fool_dict['Alg2']] * 100., '-g*', markersize=10)
    ax.plot(epsilon, X[:,fool_dict['FGS']]*100., '-ms')
    ax.plot(epsilon, X[:, fool_dict['rand']] * 100., '-^k')


    plt.xlabel('epsilon', fontsize=16)
    plt.ylabel('Fooling Ratio (in %)', fontsize=16)
    ax.grid()
    plt.ylim((-5,105))
    if legend==True:
        ax.legend(['DeepFool', 'Alg1 (10 its)', 'Alg1 (5 its)', 'Alg1', 'Alg2', 'FastGrad', 'random'], fontsize=16)
    ax.tick_params(axis='both', labelsize='large')
    plt.savefig( fname, format='eps', dpi=500 )

def main():
    args = get_arguments()

    allvars = get_all_model_variables(args)
    # Load Test Dataset
    if (allvars['model2load'] == 'fcnn') or (allvars['model2load'] == 'lenet'):
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets(allvars['data_dir'], one_hot=True)
        X = mnist.test.images
        y = mnist.test.labels

        # Free Memory
        mnist = None

    if (allvars['model2load'] == 'nin') or (allvars['model2load'] == 'densenet'):
        import cifar10
        cifar10.data_path = allvars['data_dir']
        cifar10.maybe_download_and_extract()
        X, _, y = cifar10.load_test_data()


        # Free Memory
        cifar = None

    X, y = pre_process_data(X, y, allvars['model2load'])
    X, y = collect_correctly_predicted_images(X, y, allvars)

    eps_rescale = np.max(np.abs( np.max(X.flatten()) -  np.min(X.flatten()) ))


    deepfool_norms, deepfooled_vec, deepfool_its = get_deepfool_ellenorms(X, y, allvars)
    print('DeepFool fooling ratio: '+str(np.mean(deepfooled_vec)*100)+' %')
    print('DeepFool mean epsilon: '+str(np.mean(deepfool_norms[deepfooled_vec==1])/eps_rescale))
    print('DeepFool mean iterations: ' + str(np.mean(deepfool_its[deepfooled_vec == 1])))

    if args.rho_mode:
        print('Computing performance metrics...')
        print()


        rho_1 = get_robustness_metrics(X, y, deepfool_norms, deepfooled_vec, allvars, method='rho1')
        rho_2 = get_robustness_metrics(X, y, deepfool_norms, deepfooled_vec, allvars, method='rho2')/eps_rescale
        eps99 = get_robustness_metrics(X, y, deepfool_norms, deepfooled_vec, allvars, method='eps99')/eps_rescale

        print('rho_1 = ' + str(rho_1))
        print('rho_2 = ' + str(rho_2))
        print('eps99 = ' + str(eps99))

        np.savetxt(allvars['output_dir'] + 'robustness_' + allvars['model2load'] + '_' + str(allvars['n_images']) + \
                   '_' + str(int(allvars['max_epsilon'] * 1000)) + '.csv',
                   np.array([rho_1, rho_2, eps99]),
                   delimiter=";")

    else:
        print('Computing fooling ratios...')
        print()

        epsilon = np.array(np.linspace(0.001, allvars['max_epsilon'], 10))
        fool_dict = {'epsilon': 0, 'FGS': 1, 'Alg1': 2, 'Alg2': 3, 'rand': 4, 'DeepFool': 5, 'Alg1-5':6, 'Alg1-10':7}
        fool_mtx = np.zeros([len(epsilon), len(fool_dict)])

        for i in range(len(epsilon)):
            eps = epsilon[i]*eps_rescale
            print(allvars['model2load'] + ': realization '+str(i+1)+'/'+str(len(epsilon))+'...' )

            fool_mtx[i, fool_dict['epsilon']]	= epsilon[i]
            fool_mtx[i, fool_dict['FGS']]		= get_foolratio(X, y, eps, allvars, method='FGS')
            fool_mtx[i, fool_dict['Alg1']]		= get_foolratio(X, y, eps, allvars, method='Alg1', iterations=1)
            fool_mtx[i, fool_dict['Alg2']]		= get_foolratio(X, y, eps, allvars, method='Alg2')
            fool_mtx[i, fool_dict['rand']]		= get_foolratio(X, y, eps, allvars, method='rand')
            fool_mtx[i, fool_dict['DeepFool']]	= np.mean(np.array((deepfool_norms<eps) * (deepfooled_vec==1)))
            fool_mtx[i, fool_dict['Alg1-5']]    = get_foolratio(X, y, eps, allvars, method='Alg1', iterations=5)
            fool_mtx[i, fool_dict['Alg1-10']]   = get_foolratio(X, y, eps, allvars, method='Alg1', iterations=10)

        np.savetxt(allvars['output_dir']+'fool_summary_' + allvars['model2load'] +'_' +str(allvars['n_images'])+\
                   '_'+str(int(allvars['max_epsilon']*1000))+'.csv', fool_mtx, delimiter=";")

        save_foolratio_fig(fool_mtx,
                           allvars['figs_dir'] + 'fig_'+ allvars['model2load'] +'_' +str(allvars['n_images'])+\
                           '_'+str(int(allvars['max_epsilon']*1000))+'.eps',
                           fool_dict, legend=True)

if __name__ == '__main__':
    main()