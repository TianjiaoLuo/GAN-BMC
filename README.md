## StyleGAN2-ADA BMC implementation

**Stabilizing GANs’ Training with Brownian Motion Controller**<br>
Tianjiao Luo, Ziyu Zhu, Jianfei Chen, Jun Zhu<br>
https://arxiv.org/pdf/2306.10468.pdf<br>

Abstract: *The training process of generative adversarial networks (GANs) is unstable and does not converge globally. In this paper, we examine the stability of GANs from the perspective of control theory and propose a universal higher-order noisebased controller called Brownian Motion Controller (BMC). Starting with the prototypical case
of Dirac-GANs, we design a BMC to retrieve
precisely the same but reachable optimal equilibrium. We theoretically prove that the training
process of DiracGANs-BMC is globally exponential stable and derive bounds on the rate of convergence. Then we extend our BMC to normal
GANs and provide implementation instructions
on GANs-BMC. Our experiments show that our
GANs-BMC effectively stabilizes GANs’ training
under StyleGANv2-ada frameworks with a faster
rate of convergence, a smaller range of oscillation,
and better performance in terms of FID score.*

### This implementation includes Dirac-GAN-BMC and styleGAN2-ada-BMC with controller on generator
### This repo is modified from styleGAN2-ADA-pytorch implementation 


## Note
Contact luotj21@mails.tsinghua.edu.cn if you have any questions

## Instructions on BMC
- Dirac-GAN folder includes code to reproduce figure 1 in our paper
- The Brownian Motion variable B(t) follows a Gaussian distribution
- The mean and variance of B(t) are hyperparameters to be changed 
- The controller can be added to either generator or discriminator or both depending on the nature of GAN's skeleton
- For WGAN series we add BMC on discriminator and for styleGAN we add BMC on geneator 
