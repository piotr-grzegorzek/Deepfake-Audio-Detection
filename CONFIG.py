import os

DATA_PATH = os.getcwd() + "\\data\\"  # Absolue path is required in listing sound names in builder.py modules
# Root directories of training, validation and test sound samples
TRAIN_DIR = "train"
VALID_DIR = "valid"
TEST_DIR = "test"

SOUND_LENGTH = 54812  # Length of a sound array
FREQUENCY = 16000  # Frequency of a sound (Set to None for default)
MFCC_CHANNELS = 20  # Mel's spectrogram channels

MODEL_PARAMS = {
    'num_conv_blocks': 8,
    'num_conv_filters': 32,
    'spatial_dropout_fraction': 0.05,
    'residual_con': 2,
    'num_dense_layers': 1,
    'num_dense_neurons': 50,
    'dense_dropout': 0,
    'learning_rate': 0.0001,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy']
}
EPOCHS = 10000000
BATCH_SIZE = 16
LOSS_DIFF = 0.04  # Early Stopping, max difference between valid loss and train loss
MODEL_PATH = "saved-models/model1.h5"
