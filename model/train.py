import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pyecg.io import load_data

# load data
annotated_records, samples_info = load_data("./data/train.rpeak")
print("Train data loaded, number of sampels:", str(len(samples_info)))


labels = []
for sample in samples_info:
    labels.append(sample[3])
print(len(labels))


from arch_rpeak2 import model_arch

params_model = {
    "x_input_dim": samples_info[0][2] - samples_info[0][1],  # win size
    "out_seq_len": len(labels[0]),
    "num_classes": 2,
    "regularizer": None,
}
print(params_model)
params_train = {"batch_size": 128, "epochs": 1}

mymodel = model_arch(params_model)
opt = tf.keras.optimizers.Adam(
    learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam"
)
mymodel.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    loss_weights=None,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=8, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.9,
    patience=5,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.0001,
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "checkpoint/keras.exp",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)
import time

batch_sleep = tf.keras.callbacks.LambdaCallback(
    on_batch_end=lambda batch, logs: time.sleep(0.0001)
)

callbacks = [early_stopping, model_checkpoint, reduce_lr, batch_sleep]
callbacks = [batch_sleep]

from pyecg.data_rpeak import ECGSequence

train_generator = ECGSequence(
    annotated_records,
    samples_info,
    binary=True,
    batch_size=params_train["batch_size"],
    raw=True,
    interval=36,
)

print("train_generator.shape: ", str(train_generator.__getitem__(0)[0].shape))
print(train_generator.__getitem__(0)[0].shape, train_generator.__getitem__(0)[1].shape)

# model fitting
model_history = mymodel.fit(
    generator=train_generator,
    # validation_data=validation_generator,
    use_multiprocessing=True,
    # workers=2,
    epochs=params_train["epochs"],
    verbose=1,
    callbacks=callbacks,
)
