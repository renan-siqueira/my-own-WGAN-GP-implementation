# My Own WGAN-GP Implementation

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

---

## 1. Cloning the Repository:

To clone this repository, use the following command:
```bash
git clone https://github.com/renan-siqueira/my-own-WGAN-GP-implementation.git
```

---

## 2. Creating and activating the virtual environment:

__Windows:__
```bash
python -m venv virtual_environment_name
```

To activate the virtual environment:
```bash
virtual_environment_name\Scripts\activate
```

__Linux/Mac:__
```bash
python3 -m venv virtual_environment_name
```

To activate the virtual environment:
```bash
source virtual_environment_name/bin/activate
```

---

## 3. Installing the dependencies:

__Windows / Linux / Mac:__
```bash
pip install -r requirements.txt
```

---

*__If you have a GPU, follow the steps in the "How to Use GPU" section (below). Otherwise, if you're not using a GPU, install PyTorch with the following command:__*
```bash
pip install torch torchvision torchaudio
```

---

# How to Use GPU:

## 1. Installing specific dependencies:

After creating and activating your virtual environment:

__Windows/Linux/Mac:__

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

*__Note: Make sure your hardware and operating system are compatible with CUDA 12+.__*

---

## 4. Preparing the dataset:

- 1. Create a folder named `dataset` inside the `src` folder.
- 2. Inside the `dataset` folder, create another folder with a name of your choice for the labels.
- 3. Copy all the images you wish to use for training into this folder.

---

## 5. Configuring training parameters:

The `src/json/training_params.json` file is set up with optimized parameters for this type of architecture. However, feel free to modify it according to your needs.

---

## 6. How to use the main script:

The `run.py` script is now your central point for executing various operations. It has been set up to accept arguments to dictate its behavior. Here's how to use it:

### Training the model:

To train the model, execute the following command:
```bash
python run.py --training
```

---

## 7. Monitoring the Training:

- You can follow the progress directly in the terminal or console.
- A log file will be generated in the directory specified version training.
- At the end of each epoch, samples of generated images will be saved in the folder of version training, inside the `samples` folder.

---

## 8. How to generate images after completing the training (Beta version):

To generate images after completing the training, execute:
```bash
python run.py --image
```

*You can adjust the parameters for image generation in the configuration file at `settings.PATH_IMAGE_PARAMS`.*

---

## 9. How to generate a video through interpolation of the generated images (Beta version):

To generate a video through interpolation of the generated images, execute:
```bash
python run.py --video
```

*Adjust the parameters for video generation in the configuration file located at settings.PATH_VIDEO_PARAMS.*

---

## 10. Upscaling:

If you want to upscale the generated __images__ or __video__, use the `--upscale` argument followed by the width value:
```bash
python run.py --image --upscale 1024
```

*Replace `--image` with `--video` if you're generating a video. The above command will upscale the images to a width of 1024 pixels. Adjust as needed.*
