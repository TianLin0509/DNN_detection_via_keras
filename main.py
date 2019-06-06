from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from generations import *
import tensorflow as tf


def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32))),
            1))
    return err


input_bits = Input(shape=(payloadBits_per_OFDM * 2,))
temp = BatchNormalization()(input_bits)
temp = Dense(n_hidden_1, activation='relu')(input_bits)
temp = BatchNormalization()(temp)
temp = Dense(n_hidden_2, activation='relu')(temp)
temp = BatchNormalization()(temp)
temp = Dense(n_hidden_3, activation='relu')(temp)
temp = BatchNormalization()(temp)
out_put = Dense(n_output, activation='sigmoid')(temp)
model = Model(input_bits, out_put)
model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
model.summary()
checkpoint = callbacks.ModelCheckpoint('./temp_trained_25.h5', monitor='val_bit_err',
                                       verbose=0, save_best_only=True, mode='min', save_weights_only=True)
model.fit_generator(
    training_gen(1000,25),
    steps_per_epoch=50,
    epochs=10000,
    validation_data=validation_gen(1000, 25),
    validation_steps=1,
    callbacks=[checkpoint],
    verbose=2)

model.load_weights('./temp_trained_25.h5')
BER = []
for SNR in range(5, 30, 5):
    y = model.evaluate(
        validation_gen(10000, SNR),
        steps=1
    )
    BER.append(y[1])
    print(y)
print(BER)
BER_matlab = np.array(BER)
import scipy.io as sio
sio.savemat('BER.mat', {'BER':BER_matlab})