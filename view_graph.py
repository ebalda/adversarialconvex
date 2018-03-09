"""
Description...

File: view_graph.py
Author: Emilio Balda <emilio.balda@ti.rwth-aachen.de>
Organization:  RWTH Aachen University - Institute for Theoretical Information Technology
"""

import tensorflow as tf
import argparse
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """

    # ********************* DEFAULT INPUT VARIABLES (edit if necesary) *************************
    model2load = 'fcnn'
    models_dir = 'pretrainedmodels/'
    visual_dir = 'visualization_files/'
    # ********************* ******************************************* *************************

    parser = argparse.ArgumentParser(description="Creates tensorboard visualization files for ")
    parser.add_argument("--model2load", type=str, default=model2load,
                        help="model to be loaded: either of these --> fcnn, lenet, nin, densenet. Default value = " + model2load)
    parser.add_argument("--models-dir", type=str, default=models_dir,
                        help="Path to the directory containing the pre-trained model(s). Default value = " + models_dir)
    parser.add_argument("--visual-dir", type=str, default=visual_dir,
                        help="Path to the directory where the output visualization files will be stored. Default value = " + visual_dir)
    return parser.parse_args()

def get_all_model_variables(args):
    '''
    Description...
    '''
    if (args.model2load == 'fcnn'):
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'visual_dir':args.visual_dir,
        'graph_directory':'savedmodel_fcnn_mnist/',
        'graph_file':'fcnn.ckpt.meta'
        }
    elif (args.model2load == 'lenet'):
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'visual_dir':args.visual_dir,
        'graph_directory':'savedmodel_lenet_mnist/',
        'graph_file':'lenet.ckpt.meta'
        }
    elif args.model2load == 'nin':
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'visual_dir':args.visual_dir,
        'graph_directory':'savedmodel_nin_cifar/',
        'graph_file':'nin.ckpt.meta'
        }
    elif args.model2load == 'densenet':
        modelvarnames = {
        'model2load':args.model2load,
        'models_dir':args.models_dir,
        'visual_dir':args.visual_dir,
        'graph_directory':'savedmodel_densenet_cifar/',
        'graph_file':'graph-0301-145141.meta'
        }
    else:
        print('Error:  Select a valid model (fcnn, lenet, nin, densenet)')
        modelvarnames = None

    return modelvarnames

def main():
    args = get_arguments()
    allvars = get_all_model_variables(args)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:
        saver = tf.train.import_meta_graph(allvars['models_dir'] +\
                                           allvars['graph_directory'] +\
                                           allvars['graph_file'])

        saver.restore(sess, tf.train.latest_checkpoint(allvars['models_dir'] +\
                                                       allvars['graph_directory']))
        print("Model restored.")
        print()

        file_writer = tf.summary.FileWriter(allvars['visual_dir']+allvars['model2load']+'/', sess.graph)
        print('Tensorboard Visualization on:')
        print('tensorboard --logdir='+allvars['visual_dir']+allvars['model2load']+'/')
        print()

if __name__ == '__main__':
    main()