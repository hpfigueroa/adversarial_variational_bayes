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

## Variational Autoencoder training
VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood:

$$\log p(x) \ge \text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x, z)}{q(z|x)}\right].$$

In practice, we optimize the single sample Monte Carlo estimate of this expectation:

$$\log p(x| z) + \log p(z) - \log q(z|x),$$
where $z$ is sampled from $q(z|x)$.
To compare to AVB, I extract each contribution to the ELBO loss $\log p(x| z)$, $\log p(z)$ and $\log q(z|x)$ so I can use them as metrics. 

## Variational Adversarial Bayes training
Like other adversarial models, AVB uses an additional optimizer to train the discriminative network. The training process for the encoder and decoder networks is similar to the AVB training, with the replacement of the adversary output of  similarly to the VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood:

$$\log p(x) \ge \text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x, z)}{q(z|x)}\right].$$

Which is approximated by:  

$$\log p(x| z) + \log p(z) - \log q(z|x)$$

However, since the backpropagation algorithm is not practical when $q(z|x)$ is represented by an arbitrary distribution, the term $\log p(z) - \log q(z|x)$ is replaced by the adversary output $T(x,z)$. Therefore, the ELBO loss is approximated to:  

$$\log p(x| z) - T(x,z)$$

For the adversary, the loss is defined by how well it discriminates real latent samples from model latent samples. The implementation uses this approximation:

$$\log \sigma (T(x,z_{model})) + \log (1-\sigma (T(x,z_{prior})))$$

where $z$ is sampled from $q(z|x)$
