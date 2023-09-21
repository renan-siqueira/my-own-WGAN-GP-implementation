# My Own GAN Implementation

My own GAN implementation with Pytorch.
Adapted to utilize GPU if you have an NVIDIA graphics card.

The architecture chosen for this project was WGAN-GP.

---

# How to Use This Project

## 1. Cloning the Repository:

To clone this repository, use the following command:
```bash
$ git clone https://github.com/renan-siqueira/my-own-gan-implementation.git
```

## 2. Creating and activating the virtual environment:

__Windows:__
```bash
$ python -m venv virtual_environment_name
```

To activate the virtual environment:
```bash
$ virtual_environment_name\Scripts\activate
```

__Linux/Mac:__
```bash
$ python3 -m venv virtual_environment_name
```

To activate the virtual environment:
```bash
$ source virtual_environment_name/bin/activate
```

## 3. Installing the dependencies:

__Windows / Linux / Mac:__
```bash
$ pip install -r requirements.txt
```

## 4. Preparing the dataset:

- 1. Create a folder named `dataset` at the project root.
- 2. Inside the "dataset" folder, create another folder with a name of your choice for the labels.
- 3. Copy all the images you wish to use for training into this folder.

## 5. Configuring training parameters:

The `parameters.json` file is set up with optimized parameters for this type of architecture. However, feel free to modify it according to your needs.

## 6. How to train the model:

Run the following command:
```bash
$ python main.py
```

## 7. Monitoring the Training:

- You can follow the progress directly in the terminal or console.
- A log file will be generated in the directory specified in the `parameters.json` file.
- At the end of each epoch, samples of generated images will be saved in the configured directory, inside the `samples` folder.

---

# How to Use GPU:

## 1. Installing specific dependencies:

After creating and activating your virtual environment:

__Windows/Linux/Mac:__

```bash
$ pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

*__Note: Make sure your hardware and operating system are compatible with CUDA 12+.__*
