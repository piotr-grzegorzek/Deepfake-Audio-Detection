from keras.layers import Add, Lambda, Concatenate, SpatialDropout1D, Input, Activation, Dense, Conv1D, Dropout, \
    BatchNormalization
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import Model
from CONFIG import MFCC_CHANNELS, MODEL_PARAMS


def pooling(x):
    target = x[1]
    inputs = x[0]
    mask_val = 0

    custom_mask = K.equal(inputs, mask_val)
    custom_mask = K.all(custom_mask, axis=-1, keepdims=True)

    custom_mask = 1 - K.cast(custom_mask, K.floatx())

    steps_per_sample = K.sum(custom_mask, axis=1, keepdims=False)

    target = target * custom_mask

    means = K.sum(target, axis=1, keepdims=False) / steps_per_sample
    return means


POST_MFCC_SOUND_LENGTH = None
def build_model():
    K.clear_session()
    input_dim = Input(shape=(MFCC_CHANNELS, POST_MFCC_SOUND_LENGTH), name='input_dim')

    conv_blocks_num = MODEL_PARAMS['num_conv_blocks']
    conv_filters = MODEL_PARAMS['num_conv_filters']
    spatial_dropout = MODEL_PARAMS['spatial_dropout_fraction']
    residual_num = MODEL_PARAMS['residual_con']
    dense_layers_num = MODEL_PARAMS['num_dense_layers']
    dense_neurons_num = MODEL_PARAMS['num_dense_neurons']
    dense_dropout = MODEL_PARAMS['dense_dropout']
    lr = MODEL_PARAMS['learning_rate']
    loss = MODEL_PARAMS['loss']
    metric = MODEL_PARAMS['metrics']

    convnet = []
    convnet_5 = []
    convnet_7 = []

    for ly in range(0, conv_blocks_num):
        if ly == 0:
            convnet.append(Conv1D(conv_filters, 3, strides=1, activation='linear', padding='causal')(input_dim))
            convnet_5.append(Conv1D(conv_filters, 5, strides=1, activation='linear', padding='causal')(input_dim))
            convnet_7.append(Conv1D(conv_filters, 7, strides=1, activation='linear', padding='causal')(input_dim))
        else:
            convnet.append(
                Conv1D(conv_filters * (ly * 2), 3, strides=1, activation='linear', padding='causal')(convnet[ly - 1]))
            convnet_5.append(
                Conv1D(conv_filters * (ly * 2), 5, strides=1, activation='linear', padding='causal')(convnet_5[ly - 1]))
            convnet_7.append(
                Conv1D(conv_filters * (ly * 2), 7, strides=1, activation='linear', padding='causal')(convnet_7[ly - 1]))

        convnet[ly] = LeakyReLU()(convnet[ly])
        convnet_5[ly] = LeakyReLU()(convnet_5[ly])
        convnet_7[ly] = LeakyReLU()(convnet_7[ly])
        if residual_num > 0 and (ly - residual_num) >= 0:
            res_conv = Conv1D(conv_filters * (ly * 2), 1, strides=1, activation='linear', padding='same')(
                convnet[ly - residual_num])
            convnet[ly] = Add(name=f'residual_con_3_{ly}')([convnet[ly], res_conv])
            res_conv_5 = Conv1D(conv_filters * (ly * 2), 1, strides=1, activation='linear', padding='same')(
                convnet_5[ly - residual_num])
            convnet_5[ly] = Add(name=f'residual_con_5_{ly}')([convnet_5[ly], res_conv_5])
            res_conv_7 = Conv1D(conv_filters * (ly * 2), 1, strides=1, activation='linear', padding='same')(
                convnet_7[ly - residual_num])
            convnet_7[ly] = Add(name=f'residual_con_7_{ly}')([convnet_7[ly], res_conv_7])

        if ly < (conv_blocks_num-1):
            convnet[ly] = SpatialDropout1D(spatial_dropout)(convnet[ly])
            convnet_5[ly] = SpatialDropout1D(spatial_dropout)(convnet_5[ly])
            convnet_7[ly] = SpatialDropout1D(spatial_dropout)(convnet_7[ly])

    dense = Lambda(lambda x: pooling(x))([input_dim, convnet[ly]])
    dense_5 = Lambda(lambda x: pooling(x))([input_dim, convnet_5[ly]])
    dense_7 = Lambda(lambda x: pooling(x))([input_dim, convnet_7[ly]])

    dense = Concatenate()([dense, dense_5, dense_7])

    for layers in range(dense_layers_num):
        dense = Dense(dense_neurons_num, activation='linear')(dense)
        dense = BatchNormalization()(dense)
        dense = LeakyReLU()(dense)
        dense = Dropout(dense_dropout)(dense)
    output_layer = Dense(1)(dense)
    output_layer = Activation('sigmoid')(output_layer)
    model = Model(inputs=input_dim, outputs=output_layer)
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt, loss=loss, metrics=metric)
    return model
