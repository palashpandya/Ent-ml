import string
from math import floor
from itertools import combinations
from functions import *

#Constants used
# Input for the NN (before one-hot encoding)
DIM = 2 # 1->Real part;  2-> Complex part
# List of Local Dimensions for AME(n,d)
DIM_LIST = [6,6,6,6] #
# n in AME(n,d)
NUM_SYS = len(DIM_LIST)
# number of subsystems to trace out to test if the state is AME
NUM_TROUT = NUM_SYS - floor(NUM_SYS / 2)
# Identity on the reduced subsystem for comparison
IDn = tf.eye(DIM_LIST[0]*DIM_LIST[1], dtype=tf.complex64)
tr = DIM_LIST[0]*DIM_LIST[1]

def generate_TT_data(num_pure):
    # count=0
    while True:
        x = tf.one_hot(tf.constant(range(num_pure)), depth=num_pure)
        y = np.array([tf.reshape(tf.zeros_like(x),[-1]) for _ in range(num_pure)])
        # count+=1
        yield tf.math.real(x),tf.math.real(y)

@tf.function
def make_Unitary(ypred):
    coeffs = tf.complex(ypred[0, :], ypred[1, :])
    return tf.multiply(tf.complex(1.,0.),tf.reshape(coeffs/tf.norm(coeffs) ,[np.prod(DIM_LIST[:2]),np.prod(DIM_LIST[2:])]))


@tf.function
def AME_loss(y_true, y_pred):
    """
    Define the loss function for the output of the NN to be an AME state
    :param y_true: Does nothing
    :param y_pred: output of the NN, to be restructured as a unitary matrix
    :return: loss value
    """
    unit = make_Unitary(y_pred)
    unit_pt1 = tf.reshape(tf.einsum('ijkl->ilkj', tf.reshape(unit,shape=DIM_LIST)), shape=[np.prod(DIM_LIST[:2]),np.prod(DIM_LIST[2:])])
    # unit_pt2 = tf.reshape(tf.einsum('ijkl->kjil', tf.reshape(unit, shape=DIM_LIST)),
    #                      shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    unit_re1 = tf.reshape(tf.einsum('ijkl->ikjl', tf.reshape(unit, shape=DIM_LIST)),
                         shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    # unit_re2 = tf.reshape(tf.einsum('ijkl->ljki', tf.reshape(unit, shape=DIM_LIST)),
    #                       shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    uudag = tf.matmul(unit,unit,adjoint_b=True);
    uudag = tf.scalar_mul(tr/tf.linalg.trace(uudag), uudag)
    pt1pt1dag = tf.matmul(unit_pt1, unit_pt1, adjoint_b=True)
    pt1pt1dag = tf.scalar_mul(tr/tf.linalg.trace(pt1pt1dag), pt1pt1dag)
    re1re1dag = tf.matmul(unit_re1, unit_re1, adjoint_b=True)
    re1re1dag = tf.scalar_mul(tr/tf.linalg.trace(re1re1dag), re1re1dag)
    d1 = metric_hsd(uudag, IDn)
    pt1 = metric_hsd(pt1pt1dag, IDn)
    # pt2 = metric_hsd(tf.matmul(unit_pt2, unit_pt2, adjoint_b=True), IDn)
    re1 = metric_hsd(re1re1dag, IDn)
    # re2 = metric_hsd(tf.matmul(unit_re2, unit_re2, adjoint_b=True), IDn)
    loss = d1+pt1+re1

    return loss


def early_stop():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000000001, patience=5, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=True, start_from_epoch=10)


def make_density(ypred):
    coeffs = tf.reshape(tf.complex(ypred[0, :], ypred[1, :]), [np.prod(DIM_LIST), 1])
    coeffs = tf.matmul(coeffs, coeffs, adjoint_b=True)
    # rho = Qobj(coeffs)
    # rho = rho/rho.tr()
    coeffs = tf.scalar_mul(1 / tf.linalg.trace(coeffs), coeffs)
    return coeffs

def main():
    """
    Call the appropriate functions for generating the AME state of 4 qu'd'its.
    :return: AME(4,d)
    """
    inputs = tf.one_hot(tf.constant(range(DIM)), depth=DIM)

    num_layers = len(DIM_LIST)
    layer_width = np.prod(DIM_LIST)
    layer_width_0 = np.prod(DIM_LIST)
    model = tf.keras.models.Sequential(
        [tf.keras.Input(shape=(DIM,))])
    # Hidden layers
    for i in range(int(num_layers/2)):
        model.add(tf.keras.layers.Dense(int(layer_width), activation='mish'))
    # Output layer
    # model.add(tf.keras.layers.BatchNormalization())
    # for i in range(int(num_layers / 2)):
    #     model.add(tf.keras.layers.Dense(int(layer_width), activation='mish'))

    model.add(tf.keras.layers.Dense(int(layer_width_0), activation='mish'))
    result = model.predict(inputs)
    print("Printing the model Summary")
    print(model.summary())
    print("Check if we can make a valid Unitary our of the results")
    # print(result)
    unit=make_Unitary(result)
    unit_pt = tf.reshape(tf.einsum('ijkl->ilkj', tf.reshape(unit, shape=DIM_LIST)),
                         shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    print(tf.matmul(unit,unit,adjoint_a=True))
    idn = tf.eye(DIM_LIST[0] * DIM_LIST[1],dtype=tf.complex64)
    print(f"loss A+B: A={metric_hsd(tf.matmul(unit,unit,adjoint_a=True),idn)} and B={metric_hsd(tf.matmul(unit_pt,unit_pt,adjoint_a=True),idn)}")

    optimizer = tf.optimizers.SGD(learning_rate=0.001, nesterov=True, momentum=0.9)
    model.compile(optimizer="adam", loss=AME_loss, metrics=[AME_loss])
    history = model.fit(
        generate_TT_data(DIM),
        batch_size=DIM*4,
        epochs=20,
        steps_per_epoch=200,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=generate_TT_data(DIM),
        callbacks=[early_stop()],
        validation_steps=100, class_weight=None,
        shuffle=False, initial_epoch=0
    )
    yres = model.predict(inputs)

    print("This is the result of the NN:")
    result = make_Unitary(yres)

    print(tf.matmul(result,result,adjoint_a=True))

    unit = make_Unitary(yres)
    unit_pt1 = tf.reshape(tf.einsum('ijkl->ilkj', tf.reshape(unit, shape=DIM_LIST)),
                          shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    # unit_pt2 = tf.reshape(tf.einsum('ijkl->kjil', tf.reshape(unit, shape=DIM_LIST)),
    #                      shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    unit_re1 = tf.reshape(tf.einsum('ijkl->ikjl', tf.reshape(unit, shape=DIM_LIST)),
                          shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])
    # unit_re2 = tf.reshape(tf.einsum('ijkl->ljki', tf.reshape(unit, shape=DIM_LIST)),
    #                       shape=[np.prod(DIM_LIST[:2]), np.prod(DIM_LIST[2:])])

    d1 = metric_hsd(tf.matmul(unit, unit, adjoint_b=True), IDn)
    pt1 = metric_hsd(tf.matmul(unit_pt1, unit_pt1, adjoint_b=True), IDn)
    # pt2 = metric_hsd(tf.matmul(unit_pt2, unit_pt2, adjoint_b=True), IDn)
    re1 = metric_hsd(tf.matmul(unit_re1, unit_re1, adjoint_b=True), IDn)
    print(d1,pt1,re1)

    result = make_density(yres)


    io.mmwrite(f"data/AME_{len(DIM_LIST)}_{DIM_LIST[0]}_u_constr.mtx", result)


if __name__ == '__main__':
    main()
