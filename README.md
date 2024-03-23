# Deepfake Audio Detection

This project focuses on solving the increasing problem of deepfakes, specifically in the audio domain. It uses convolutional neural networks to detect whether a given audio sample is real or a deepfake. The solution was trained on the AVspoof database.

# Project Structure
The project is structured as follows:

- `main.py`: This is the main module that controls the entire project.
- `CONFIG.py`: This module contains the configuration parameters for the project.
- `callbacks.py`: This module contains the Keras callbacks for Model Checkpoint and Early Stopping.
- `labels/`: This directory contains the labels for the audio samples.
- `data/`: This directory contains the audio samples.

# Usage
To use this project, first ensure that your parameters in `CONFIG.py` are correct. The parameters you should check are `TRAIN_DIR`, `VALID_DIR`, and `TEST_DIR`. These directories should contain your audio files.

The `main.py` module contains the `main()` function which controls the whole project. Here are some functions you can use in the `main()` function:

- `build_labels_from(dir_name)`: This function reads sound names from the `dir_name` parameter in the correct order. It then checks if the sound is a spoof or bonafide and appends the value to a list.
- `read_labels_from(txt_file_name)`: This function reads a text file located in the `labels/clean` directory created in the `build_labels_from` function.
- `build_data_from(dir_name)`: This function reads sounds from the `dir_name` parameter and applies MFCC sampling and Min Max function.

# Requirements
To install the project requirements, run the following command:

```sh
pip install -r requirements.txt
```