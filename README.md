# My Own GAN Implementation

My own GAN implementation with Pytorch. Adapted to use GPU if you have an NVIDIA graphic card.

The architecture chosen for this project was WGAN-GP.

## A brief explanation of WGAN:

WGAN (Wasserstein Generative Adversarial Network) is a variant of generative adversarial networks (GANs) that uses the Wasserstein distance to measure the difference between the real data distribution and the generated distribution. This approach was introduced to address the convergence issues of traditional GANs, providing more stable training.

## A brief explanation of WGAN-GP:

WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) is an extension of WGAN. The "GP" refers to gradient penalty, a term added to the loss function to ensure gradients are bounded and to avoid the "mode collapse" issue often seen in GANs. This penalty forces the discriminator to have gradients of norm close to 1, which helps ensure smoother and more stable training.

## Comparison between the two architectures:

When comparing WGAN and WGAN-GP, the main reason to opt for WGAN-GP is the introduction of the Gradient Penalty, in the original WGAN, to ensure the critic function (also known as discriminator) is Lipschitz continuous, one has to clip the weights, a process known as "weight clipping". However, this method can lead to optimization issues and networks that are not expressive enough.

WGAN-GP, on the other hand, introduces a penalty on the gradient of the critic function, ensuring it's Lipschitz continuous without the need for weight clipping. This results in more stable training, avoiding the problems associated with weight clipping. In short, by using WGAN-GP over WGAN, one benefits from smoother and more consistent training and a potentially more expressive model.

## What is "Lipschitz continuous"?

A function is said to be Lipschitz continuous if there exists a constant "L" (known as the Lipschitz constant) such that the absolute difference between the function values at any two points is bounded by the product of that constant and the distance between those points.

The Lipschitz condition ensures that the function does not have very abrupt oscillations, meaning it isn't too "steep" over any interval.

## Why WGAN-GP?

Training traditional GANs (Generative Adversarial Networks) often faces stability issues, which can result in low-quality image generation or model convergence failures WGAN, with is Wasserstein distance-based approach, brought significant improvements in terms of training stability. However, the need for weight clipping in the original WGAN can limit the network's capacity.

Enter WGAN-GP, which introduces the gradient penalty. By replacing weight clipping with the gradient penalty, WGAN-GP ensures the critic function remains Lipschitz continuous without constraining the network's expressive power. This tweak improves training stability and helps yield higher-quality outputs.

If you're looking for a GAN approach that combines robust training with the capability of producing high-quality outputs, WGAN-GP is undoubtedly an excellent choice.
