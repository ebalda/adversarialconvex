# Generating Adversarial Examples using Convex Programming in tensorflow
This repository contains the tensorflow implementation our paper [here](http://arxiv.org/).

Citation:

     @inproceedings{balda2017,
          title={something},
          author={Balda },
          booktitle={Axiv},
          year={2017}
      }

## Dependencies:

+ Python 3
+ TensorFlow >= 1.4

## Compute the fooling ratio

```
python main.py --model2load=fcnn --n-images=2000 --max-epsilon=0.1
```
![figrues](fig_fcnn_1024_100.eps)

## Questions?

Please drop [me](http://www.cs.cornell.edu/~yli) a line if you have any questions!