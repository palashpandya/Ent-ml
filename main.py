import numpy as np
import tensorflow as tf
from scipy import io

target  = tf.squeeze(tf.constant([io.mmread('target.mtx')]))

from functions import *


if __name__ == '__main__':
    # Build separable approximation of target state:
    # global target
    target = tf.cast(target,dtype=tf.complex128)

    # 3 qubit W state
    target =  tf.constant([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype='complex128')
    target = tf.constant([[0.5, 0, 0, 0, 0, 0, 0, 0.5],
                    [0, 0., 0., 0, 0., 0, 0, 0],
                    [0, 0., 0., 0, 0., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0., 0., 0, 0., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0.5, 0, 0, 0, 0, 0, 0, 0.5]], dtype='complex128')
    # target = tf.scalar_mul(tf.cast(tf.complex(0.19,0.),dtype=tf.complex128), tf.constant([[0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
    #                 [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0]], dtype='complex128'))+ tf.scalar_mul(tf.cast(tf.complex((1-0.19)*0.125,0.),dtype=tf.complex128), tf.eye(8,8,dtype=tf.complex128))
    # # Bell state 2 qubit
    # target = tf.constant([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype='complex128')
    # print(target)
    global num_pure
    num_pure = 32
    inputs = tf.one_hot(tf.constant(range(num_pure)),depth=num_pure)
    # print(inputs)

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(num_pure,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(13,activation='tanh')
        ]
    )
    result = model.predict(inputs)
    print("Printing the model Summary")
    print(model.summary())
    # print("Check if the sum of the outputs is 1:")
    # print(tf.reduce_sum(result[1]))
    # print("Check if we can make a valid density our of the results")
    # print(make_density(result))
    model.compile(optimizer='adam',loss = custom_loss,metrics=[custom_loss])
    batchsize =num_pure
    history = model.fit(
        generate_test_train_XY(target,num_pure),
        batch_size=num_pure*2,
        epochs=4,
        steps_per_epoch=num_pure**2,
        validation_data= generate_test_train_XY(target,num_pure),
        callbacks = [callback()],
        validation_steps = 32, class_weight = None,
        shuffle = False, initial_epoch = 0
    )
    yres = model.predict(inputs)

    print("This is the result of the NN:")
    print(make_density3q(yres))
    io.mmwrite("data/W3_PPT_CSS.mtx", make_density3q(yres))
    verify_density_matrix(make_density3q(yres).numpy())
    # print("dist of css with half noise + W: ", metric_hsd(target2,make_density3q(yres)))

    css = make_density3q(yres)

    print(f"{tf.sqrt(metric_hsd(css,target))} is the final distance")


    # print(inputs)