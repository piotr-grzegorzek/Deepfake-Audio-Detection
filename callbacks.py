import keras
from CONFIG import MODEL_PATH, LOSS_DIFF


class LossHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        check = self.val_losses[-1] - self.losses[-1]
        if check >= LOSS_DIFF:
            self.model.stop_training = True


ES = LossHistory()
MCH = keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=True, mode='auto', period=1)
