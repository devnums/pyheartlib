# Train the example model
import os

from pyheartlib.extra.utils import reset_seed
from pyheartlib.io import load_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402

cdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cdir)
print("Current directory changed to:\n", cdir)
data_dir = "data"
train_data = os.path.join(data_dir, "train.rpeak")
val_data = os.path.join(data_dir, "val.rpeak")
mdl_checkpoint = os.path.join(os.getcwd(), "checkpoint/keras.exp")
interval_value = 6

reset_seed()

# load train data
annotated_records, samples_info = load_data(train_data)
print("Train data loaded, number of sampels:", str(len(samples_info)))

labels = []
for sample in samples_info:
    labels.append(sample[3])
print(len(labels))

# import model architecture
from model_arch import model_arch  # noqa: E402

params_model = {
    "x_input_dim": samples_info[0][2] - samples_info[0][1],  # win size
    "out_seq_len": len(labels[0]),
    "num_classes": 2,
    "regularizer": None,
}
print(params_model)
params_train = {"batch_size": 32, "epochs": 3}

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
    mdl_checkpoint,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)
import time  # noqa: E402

batch_sleep = tf.keras.callbacks.LambdaCallback(
    on_batch_end=lambda batch, logs: time.sleep(0.0001)
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]
# callbacks = [batch_sleep]

from pyheartlib.data_rpeak import ECGSequence  # noqa: E402

trainseq = ECGSequence(
    annotated_records,
    samples_info,
    binary=True,
    batch_size=params_train["batch_size"],
    raw=True,
    interval=interval_value,
)

print("trainseq.shape: ", str(trainseq[0][0].shape))
print(trainseq[0][0].shape, trainseq[0][1].shape)

# load validation data
annotated_records_val, samples_info_val = load_data(val_data)

validationseq = ECGSequence(
    annotated_records_val,
    samples_info_val,
    binary=True,
    batch_size=params_train["batch_size"],
    raw=True,
    interval=interval_value,
)

# train model
model_history = mymodel.fit(
    x=trainseq,
    validation_data=validationseq,
    use_multiprocessing=True,
    # workers=2,
    epochs=params_train["epochs"],
    verbose=1,
    callbacks=callbacks,
)
