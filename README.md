# My Own GAN Implementation

My own GAN implementation with Pytorch.
Adapted to utilize GPU if you have an NVIDIA graphics card.

The architecture chosen for this project was WGAN-GP.

## A brief explanation about WGAN:

WGAN (Wasserstein Generative Adversarial Network) is a variant of generative adversarial networks (GANs) that uses the Wasserstein distance to measure the difference between the real data distribution and the generated distribution. This approach was introduced to address the convergence issues of traditional GANs, providing more stable training.

## A brief explanation about WGAN-GP:

WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty) is an extension of WGAN. The "GP" refers to the gradient penalty, a term added to the loss function to ensure that gradients are bounded and to prevent the "mode collapse" problem often seen in GANs. This penalty forces the discriminator to have gradients with norms close to 1, helping ensure smoother and more stable training.

## Comparison between the two architectures:

When comparing WGAN and WGAN-GP, the main reason to choose WGAN-GP is the introduction of the Gradient Penalty. In the original WGAN, to ensure the critic function (also known as the discriminator) is Lipschitz continuous, it's necessary to clip the weights, a process known as "weight clipping". However, this method can lead to optimization issues and networks that aren't expressive enough.

WGAN-GP, on the other hand, introduces a penalty on the gradient of the critic function, ensuring it's Lipschitz continuous without the need to clip weights. This results in more stable training, avoiding problems associated with "weight clipping". In summary, by using WGAN-GP instead of WGAN, you benefit from smoother, more consistent training and a potentially more expressive model.

## What is "Lipschitz continuous"?

A function is said to be Lipschitz continuous if there exists a constant "L" (known as the Lipschitz constant) such that the absolute difference between the function values at any two points is bounded by the product of that constant and the distance between those points.

The Lipschitz condition ensures that the function doesn't have very abrupt oscillations, meaning it's not too "steep" over any interval.

## Why WGAN-GP?

Training traditional GANs (Generative Adversarial Networks) often faces stability issues, which can result in the generation of low-quality images or model convergence failures. WGAN, with its approach based on the Wasserstein distance, brought significant improvements in training stability. However, the need for "weight clipping" in the original WGAN can limit the network's capacity.

Enter WGAN-GP, introducing the gradient penalty. By replacing "weight clipping" with gradient penalty, WGAN-GP ensures that the critic function remains Lipschitz continuous without restricting the network's expressive power. This modification improves training stability and helps produce higher-quality outputs.

If you're looking for a GAN approach that combines robust training with the ability to produce high-quality outputs, WGAN-GP is undoubtedly an excellent choice.

# How to Use This Project

## 1. Cloning the Repository:

To clone this repository, use the following command:

```git clone https://github.com/renan-siqueira/my-own-gan-implementation.git```

## 2. Creating and activating the virtual environment:

### Windows:
```python -m venv virtual_environment_name```

To activate the virtual environment:
```virtual_environment_name\Scripts\activate```

### Linux/Mac:
```python3 -m venv virtual_environment_name```

To activate the virtual environment:
```source virtual_environment_name/bin/activate```

## 3. Installing the dependencies:

Windows / Linux / Mac:
```pip install -r requirements.txt```

## 4. Preparing the dataset:

- 1. Create a folder named "dataset" at the project root.
- 2. Inside the "dataset" folder, create another folder with a name of your choice for the labels (e.g., "images").
- 3. Copy all the images you wish to use for training into this folder.

## 5. Configuring training parameters:

The "parameters.json" file is set up with optimized parameters for this type of architecture. However, feel free to modify it according to your needs.

## 6. How to train the model:

Run the following command:

```python main.py```

## 7. Monitoring the Training:

- You can follow the progress directly in the terminal or console.
- A log file will be generated in the directory specified in the parameters.json file.
- At the end of each epoch, samples of generated images will be saved in the configured directory, inside the "samples" folder.

# How to Use GPU:

## 1. Installing specific dependencies:

After creating and activating your virtual environment:

Windows/Linux/Mac

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121```

Note: Make sure your hardware and operating system are compatible with CUDA 12+.
