import string
from math import floor

import numpy as np
from itertools import combinations
import tensorflow as tf

# from qutip.core.cy.qobjevo import Qobj
from scipy import io
# from qutip import *
# from main import metric_hsd, verify_density_matrix


NUM_PURE = 2
DIM_LIST = [3,3,3,3]
NUM_SYS = len(DIM_LIST)
NUM_TROUT = NUM_SYS - floor(NUM_SYS / 2)
print(NUM_SYS,NUM_TROUT,NUM_PURE)
IDn = tf.scalar_mul(DIM_LIST[0] ** (-NUM_SYS + NUM_TROUT), tf.eye(DIM_LIST[0] ** (NUM_SYS - NUM_TROUT)))



def verify_density_matrix(rho, tolerance=1e-9):
    """
    Verifies if a given matrix satisfies the properties of a density matrix.

    Properties checked:
    1. Hermitian: rho == rho^dagger
    2. Unit Trace: Tr(rho) == 1
    3. Positive Semi-Definite: All eigenvalues >= 0
    4. Purity: Tr(rho^2) (Check if pure or mixed)
    """
    print("-" * 30)
    print("Verifying Density Matrix Properties:")

    # Check 1: Hermitian
    is_hermitian = np.allclose(rho, rho.conj().T, atol=tolerance)
    print(f"1. Hermitian: {is_hermitian}")

    # Check 2: Unit Trace
    trace_val = np.trace(rho)
    is_unit_trace = np.isclose(trace_val, 1.0, atol=tolerance)
    print(f"2. Unit Trace: {is_unit_trace} (Trace = {trace_val:.4f})")

    # Check 3: Positive Semi-Definite
    eigenvalues = np.linalg.eigvalsh(rho)
    min_eig = np.min(eigenvalues)
    is_positive = min_eig > -tolerance
    print(f"3. Positive Semi-Definite: {is_positive}")
    print(f"   (Min Eigenvalue: {min_eig:.4e})")

    # Check 4: Purity
    purity = np.real(np.trace(rho @ rho))
    is_pure_state = np.isclose(purity, 1.0, atol=tolerance)
    state_type = "Pure State" if is_pure_state else "Mixed State"
    print(f"4. Purity Tr(rho^2): {purity:.4f} -> {state_type}")
    print("-" * 30)


def metric_hsd(r1, r2):
    # r21 = make_density(r2)
    # return tf.linalg.trace(tf.linalg.matmul((r1-r21), (r1-r21), adjoint_a = True))
    r11 = tf.cast(r1, dtype=tf.complex128)
    r22 = tf.cast(r2, dtype=tf.complex128)
    return tf.linalg.norm(r11-r22)

def partial(rho, tout, dims):
    num_indices = 2 * NUM_SYS
    indices = list(string.ascii_lowercase)[:num_indices]
    for sys in tout:
        indices[sys+NUM_SYS] = indices[sys]
    rho1 = tf.reshape(rho, DIM_LIST + DIM_LIST)
    newdims = [dims[i] for i in range(len(dims)) if i not in tout]
    return tf.reshape(tf.einsum(''.join(indices),rho1),[tf.reduce_prod(newdims),tf.reduce_prod(newdims)])


# # Example: 2 qubits, trace out second qubit
# rho = tf.constant([[1., 0., 0., 0.],
#                    [0., 0., 0., 0.],
#                    [0., 0., 0., 0.],
#                    [0., 0., 0., 0.]], dtype=tf.complex64)
#
# rho_A = partial_trace(rho, keep=[0], dims=[2, 2])
# print(rho_A.numpy())


# @tf.function
def make_density2q(ypred):
    coeffs = tf.reshape(tf.complex(ypred[0, :], ypred[1, :]), [np.prod(DIM_LIST), 1])
    coeffs = tf.matmul(coeffs, coeffs, adjoint_b=True)
    # rho = Qobj(coeffs)
    # rho = rho/rho.tr()
    coeffs = tf.scalar_mul(1 / tf.linalg.trace(coeffs), coeffs)
    return coeffs


@tf.function
def pt_loss(y_true, y_pred):
    rho = make_density2q(y_pred)
    loss = 0.
    comb_sys = combinations(range(len(DIM_LIST)), NUM_TROUT)
    for sys in comb_sys:
        loss += metric_hsd(partial(rho, tout=sys, dims=DIM_LIST), IDn)
    return loss
    # rhoa = partial(rho, tout=[1,0], dims=DIM_LIST)
    # rhob = partial(rho, tout=[0,2], dims=DIM_LIST)
    # rhoc = partial(rho, tout=[2,1], dims=DIM_LIST)
    # loss = metric_hsd(id1, rhob) + metric_hsd(id1, rhoc) + metric_hsd(rhoa, id1)



def generate_test_train_XY(state, num_pure):
    # count=0
    while True:
        x = tf.one_hot(tf.constant(range(num_pure)), depth=num_pure)
        y = np.array([tf.reshape(tf.zeros([np.prod(DIM_LIST),1]), [-1]) for _ in range(num_pure)])
        # count+=1
        yield tf.math.real(x), tf.math.real(y)


def CC_mindelta():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000000001, patience=10, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=True)#, start_from_epoch=64)


if __name__ == '__main__':
    # Build a 2 qubit AME state:

    # # Bell state 2 qubit
    target = tf.constant([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype='complex128')

    inputs = tf.one_hot(tf.constant(range(NUM_PURE)), depth=NUM_PURE)
    # print(inputs)
    num_layers = 5
    layer_width = np.prod(DIM_LIST)*2
    layer_width_0 = np.prod(DIM_LIST)
    #Input layer
    model = tf.keras.models.Sequential(
        [ tf.keras.Input(shape=(NUM_PURE,)) ])
    #Hidden layers
    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(layer_width, activation='relu'))
    #Output layer
    model.add(tf.keras.layers.Dense(np.prod(DIM_LIST), activation='tanh'))

    result = model.predict(inputs)
    print("Printing the model Summary")
    print(model.summary())
    print("Check if we can make a valid density our of the results")
    print(make_density2q(result))
    print(verify_density_matrix(make_density2q(result).numpy()))
    model.compile(optimizer='nadam', loss=pt_loss, metrics=[pt_loss])
    batchsize = NUM_PURE
    history = model.fit(
        generate_test_train_XY(target, NUM_PURE),
        batch_size=NUM_PURE * 2,
        epochs=256,
        steps_per_epoch=np.prod(DIM_LIST)*10,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=generate_test_train_XY(target, NUM_PURE),
        callbacks=[CC_mindelta()],
        validation_steps=np.prod(DIM_LIST), class_weight=None,
        shuffle=False, initial_epoch=0
    )
    yres = model.predict(inputs)

    print("This is the result of the NN:")
    print(make_density2q(yres))
    result = make_density2q(yres)



    io.mmwrite("AME_4_4.mtx", result)
    # verify_density_matrix(make_density3q(yres).numpy())
    # # print("dist of css with half noise + W: ", metric_hsd(target2,make_density3q(yres)))
    #
    # css = make_density3q(yres)
    #
    # print(f"{tf.sqrt(metric_hsd(css,target))} is the final distance")
    #
    #
    # # print(inputs)