import numpy as np
from CONFIG import TRAIN_DIR, VALID_DIR, TEST_DIR, MFCC_CHANNELS, EPOCHS, BATCH_SIZE, MODEL_PATH
from labels.builder import build_labels_from, read_labels_from
from data.builder import build_data_from
import model
from callbacks import ES, MCH


def main():
    y_t = read_labels_from(TRAIN_DIR)
    x_t = build_data_from(TRAIN_DIR)

    y_v = read_labels_from(VALID_DIR)
    x_v = build_data_from(VALID_DIR)
    train_on(np.array(x_t), np.array(y_t), np.array(x_v), np.array(y_v))

    test_labels = read_labels_from(TEST_DIR)
    test_data = build_data_from(TEST_DIR)
    test_on(test_data, test_labels)


def train_on(x_t, y_t, x_v, y_v):
    train_model = model.build_model()
    train_model.fit(x_t, y_t, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(x_v, y_v),
                    shuffle=True, verbose=2, callbacks=[ES, MCH])


def test_on(test_data, test_labels):
    predicted_well = predicted_bad = fake_total = fake_good = real_total = real_good = 0
    test_model = model.build_model()
    test_model.load_weights(MODEL_PATH)

    for index in range(0, len(test_labels)):
        sample = np.reshape(test_data[index], (1, MFCC_CHANNELS, model.POST_MFCC_SOUND_LENGTH))
        prediction = test_model.predict(sample)
        if prediction >= 0.5:
            if test_labels[index] == 1:
                predicted_well += 1
                fake_good += 1
                fake_total +=1
            else:
                predicted_bad += 1
                real_total += 1
        else:
            if test_labels[index] == 0:
                predicted_well += 1
                real_good += 1
                real_total += 1
            else:
                predicted_bad += 1
                fake_total += 1
    # print("good predictions: " + str(predicted_well)) print("bad predictions: " + str(predicted_bad))
    print("Accuracy: " + str(float(predicted_well) / float(predicted_well + predicted_bad)))
    print("Fake total: " + str(fake_total))
    print("Fake good: " + str(fake_good))
    print("Real total: " + str(real_total))
    print("Real good: " + str(real_good))


if __name__ == "__main__":
    main()
