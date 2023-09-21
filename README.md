# My Own GAN Implementation

My own GAN implementation with Pytorch.
Adapted to utilize GPU if you have an NVIDIA graphics card.

The architecture chosen for this project was WGAN-GP.

---

## With this project you will be able to:

- Train your own GAN with the images of your choice;
- Generate as many images as you want after completing the training (Beta version);
- Produce videos through interpolation of the generated images (Beta version);

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

## 8. How to generate images after completing the training (Beta version):

Open the `generate.py` file, locate the portion of the code that calls the `main` function (end of the file), and modify the variables to reflect your scenario:
```python
    train_version = 'v1' # Set the version you wish to use to generate the images
    output_directory = f'images_generated/{train_version}/' # Only change the beginning of the path to a folder where you wish to save the images
    num_samples = 4 # Number of images you want to generate

```

Execute this file with command:
```bash
$ python generate.py
```

## 9. How to generate a video through interpolation of the generated images (Beta version):

Open the `interpolate.py` file, locate the portion of the code that calls the `main` function (end of the file), and modify the variables to reflect your scenario:
```python
    train_version = 'v1' # Set the version you wish to use to generate the images

    interpolate_points = 10 # Specify how many images you wish to interpolate
    steps_between = 30 # Specify how many images you wish to generate between each interpolation
    fps = 30 # Specify how many frames per second the video should have

    video_name = 'video.avi' # Specify the name of the output video (keep the .avi extension)
```

Execute this file with command:
```bash
$ python interpolate.py
```

---

# How to Use GPU:

## 1. Installing specific dependencies:

After creating and activating your virtual environment:

__Windows/Linux/Mac:__

```bash
$ pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

*__Note: Make sure your hardware and operating system are compatible with CUDA 12+.__*
