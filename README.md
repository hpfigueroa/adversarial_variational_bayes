# adversarial_variational_bayes
Code to compare the performance between a vanilla Variational Autoencoder (VAE) model and an Adversarial Variational Bayes (AVB) model
It contains tensorflow 2.x implementations for both models and compares their performance with 4 pixel and 9 pixel synthetic datasets, based on the generative model example from:

@INPROCEEDINGS{Mescheder2017ICML,
  author = {Lars Mescheder and Sebastian Nowozin and Andreas Geiger},
  title = {Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2017}
}

The code takes some implementation ideas from the following sources:

Tensorflow tutorials:  
https://www.tensorflow.org/tutorials/generative/cvae  
https://www.tensorflow.org/tutorials/generative/dcgan  
https://www.tensorflow.org/guide/keras/functional  

Author tensorflow 1.x based repo:  
https://github.com/LMescheder/AdversarialVariationalBayes

AVB pytorch implementation post:  
https://chrisorm.github.io/AVB-pyt.html

