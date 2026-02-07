import string
from math import floor
from itertools import combinations
from functions import *

#Constants used
# Input for the NN (after one-hot encoding)
NUM_PURE = 2
# List of Local Dimensions for AME(n,d)
DIM_LIST = [5,5,5,5] #
# n in AME(n,d)
NUM_SYS = len(DIM_LIST)
# number of subsystems to trace out to test if the state is AME
NUM_TROUT = NUM_SYS - floor(NUM_SYS / 2)
# Identity on the reduced subsystem for comparison
IDn = tf.scalar_mul(DIM_LIST[0] ** (-NUM_SYS + NUM_TROUT), tf.eye(DIM_LIST[0] ** (NUM_SYS - NUM_TROUT)))



def partial(rho, tout, dims):
    """
    takes the partial trace of rho over subsystems in tout. rho has local dimensions specified in dims.
    :param rho:
    :param tout:
    :param dims:
    :return:
    """
    num_indices = 2 * NUM_SYS
    indices = list(string.ascii_lowercase)[:num_indices]
    for sys in tout:
        indices[sys+NUM_SYS] = indices[sys]
    rho1 = tf.reshape(rho, DIM_LIST + DIM_LIST)
    newdims = [dims[i] for i in range(len(dims)) if i not in tout]
    return tf.reshape(tf.einsum(''.join(indices),rho1),[tf.reduce_prod(newdims),tf.reduce_prod(newdims)])

@tf.function
def pt_loss(y_true, y_pred):
    rho = make_density2q(y_pred)
    loss = 0.
    comb_sys = combinations(range(len(DIM_LIST)), NUM_TROUT)
    for sys in comb_sys:
        loss += (metric_hsd(partial(rho, tout=sys, dims=DIM_LIST), IDn))
    return (loss)

@tf.function
def make_density2q(ypred):
    coeffs = tf.reshape(tf.complex(ypred[0, :], ypred[1, :]), [np.prod(DIM_LIST), 1])
    coeffs = tf.matmul(coeffs, coeffs, adjoint_b=True)
    # rho = Qobj(coeffs)
    # rho = rho/rho.tr()
    coeffs = tf.scalar_mul(1 / tf.linalg.trace(coeffs), coeffs)
    return coeffs

@tf.function
def make_density_real(ypred):
    coeffs = tf.reshape(ypred[0, :], [np.prod(DIM_LIST), 1])
    coeffs = tf.matmul(coeffs, coeffs, adjoint_b=True)
    # rho = Qobj(coeffs)
    # rho = rho/rho.tr()
    coeffs = tf.scalar_mul(1 / tf.linalg.trace(coeffs), coeffs)
    return coeffs

    # rhoa = partial(rho, tout=[1,0], dims=DIM_LIST)
    # rhob = partial(rho, tout=[0,2], dims=DIM_LIST)
    # rhoc = partial(rho, tout=[2,1], dims=DIM_LIST)
    # loss = metric_hsd(id1, rhob) + metric_hsd(id1, rhoc) + metric_hsd(rhoa, id1)



def generate_test_train_XY( num_pure):
    # count=0
    while True:
        x = tf.one_hot(tf.constant(range(num_pure)), depth=num_pure)
        y = np.array([tf.reshape(tf.zeros([np.prod(DIM_LIST),1]), [-1]) for _ in range(num_pure)])
        # count+=1
        yield tf.math.real(x), tf.math.real(y)


def CC_mindelta():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000000001, patience=5, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=True, start_from_epoch=1)


if __name__ == '__main__':

    # # Bell state 2 qubit
    target = tf.constant([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype='complex128')

    inputs = tf.one_hot(tf.constant(range(NUM_PURE)), depth=NUM_PURE)
    # print(inputs)
    num_layers = len(DIM_LIST)-3
    layer_width = np.prod(DIM_LIST)
    print(f"layer width is {int(layer_width)} and the data type is {type(layer_width)}")
    layer_width_0 = np.prod(DIM_LIST)
    #Input layer
    model = tf.keras.models.Sequential(
        [ tf.keras.Input(shape=(NUM_PURE,)) ])
    #Hidden layers
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(int(layer_width), activation='selu'))
    #Output layer
    model.add(tf.keras.layers.Dense(int(np.prod(DIM_LIST)), activation='selu'))

    # result = model.predict(inputs)
    # print("Printing the model Summary")
    # print(model.summary())
    # print("Check if we can make a valid density our of the results")
    # print(make_density2q(result))
    # print(verify_density_matrix(make_density2q(result).numpy()))
    optimizer = tf.optimizers.SGD(learning_rate=0.01)#,nesterov=True,momentum=0.9)
    model.compile(optimizer="adadelta", loss=pt_loss, metrics=[pt_loss])
    history = model.fit(
        generate_test_train_XY( NUM_PURE),
        batch_size=NUM_PURE * 2,
        epochs=100,
        steps_per_epoch=np.prod(DIM_LIST)*4,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=generate_test_train_XY( NUM_PURE),
        callbacks=[CC_mindelta()],
        validation_steps=np.prod(DIM_LIST), class_weight=None,
        shuffle=False, initial_epoch=0
    )
    yres = model.predict(inputs)

    print("This is the result of the NN:")
    print(make_density2q(yres))
    result = make_density2q(yres)



    io.mmwrite(f"data/AME_{len(DIM_LIST)}_{DIM_LIST[0]}.mtx", result)
    # verify_density_matrix(make_density3q(yres).numpy())
    # # print("dist of css with half noise + W: ", metric_hsd(target2,make_density3q(yres)))
    #
    # css = make_density3q(yres)
    #
    # print(f"{tf.sqrt(metric_hsd(css,target))} is the final distance")
    #
    #
    # # print(inputs)