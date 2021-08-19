import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import datetime
import pandas as pd
import os

def test():
    # 0, 0.1, 0.2, .... 5, 5.1, ..... 9.8, 9.9
    x = np.arange(start=0, stop=10, step=0.1)
    f = np.sin
    #y = np.sin(x)
    y = f(x)

    plt.plot(x, y)
    plt.savefig("sin.png")
    plt.show()

def make_model():
    inputs = tf.keras.Input(shape=(1,))
    x = layers.Dense(256)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model

def get_data(start=-10, stop=10, rate=0.8, step=0.1, f=np.sin):
    x = np.arange(start=start, stop=stop, step=step)
    size = len(x)
    mask = np.random.choice(size, size=int(size*rate))
    x_train = x[mask]
    x_test = []
    for i in x:
        if not i in x_train:
            x_test.append(i)
    x_test = np.array(x_test)

    y_train = f(x_train)
    y_test = f(x_test)

    return (x_train, x_test, y_train, y_test)


def plot(model, range=10):
    """
    x_train, x_test, y_train, y_test = get_data(start=-range, stop=-range, rate=1.0)
    x = x_train
    """
    x = np.arange(0, 9, step=0.001)
    y1 = model.predict(x)
    y2 = np.sin(x)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    

def main():
    model = make_model()

    x_train, x_test, y_train, y_test = get_data(start=0, stop=10, step=0.5)

    mse_loss = tf.keras.losses.MeanSquaredError()
    adam_optimizer = tf.keras.optimizers.Adam()
    mae_metrics = tf.keras.metrics.MeanAbsoluteError()

    model.compile(optimizer=adam_optimizer, loss=mse_loss, metrics=mae_metrics)

    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + time_str
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tensorboard --logdir logs/fit
    # で確認

    batch_size = 32
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, callbacks=tensorboard_callback, epochs=2000)

    weights_dir = "weights/" + time_str
    try:
        os.makedirs(weights_dir)
    except Exception as e:
        print(e)

    pd.to_pickle(model.get_weights(), weights_dir + "/weights.pkl")

    results = model.evaluate(x=x_test, y=y_test, callbacks=tensorboard_callback)

    print("test loss(mse), test loss(mae):", results)

    plot(model)

    
def load_model():
    model = make_model()
    path = "./weights/20210818-001628/"
    weights = pd.read_pickle(path + "/weights.pkl")

    model.set_weights(weights)
    plot(model, range=20)


if __name__ == "__main__":
    #main()
    model = make_model()
    