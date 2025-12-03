import numpy as np
import tensorflow as tf



def metric_hsd(r1,r2):
    # r21 = make_density(r2)
    # return tf.linalg.trace(tf.linalg.matmul((r1-r21), (r1-r21), adjoint_a = True))
    r11 = tf.cast(r1,dtype=tf.complex64)
    r22 = tf.cast(r2, dtype=tf.complex64)
    return tf.math.real(tf.linalg.trace(tf.linalg.matmul((r11 - r22), (r11 - r22), adjoint_a=True)))

# @tf.function
def make_density2q(ypred):
    #make the probability vector and normalize it
    pr = [ypred[i,0] for i in range(num_pure)]
    pr = tf.nn.softmax(pr)
    # make empty matrix where the CSS will be made
    part = tf.zeros_like(target)

    for i in range(num_pure): # for each vector in pure state decomp
        m = ypred[i,1:9]
        m=tf.reshape(m,shape=[2,2,2])
        r1 = tf.reshape(tf.complex( m[0,:,0], m[0,:,1] ),shape=[2,1])
        r2 = tf.reshape(tf.complex( m[1,:,0], m[1,:,1]),shape=[2,1])
        r1 = tf.matmul(r1,r1,adjoint_b=True) # pure state on subsytem A
        r1 = tf.scalar_mul(1./tf.linalg.trace(r1), r1)
        r2 = tf.matmul(r2, r2, adjoint_b=True)
        r2 = tf.scalar_mul(1. / tf.linalg.trace(r2), r2) # pure state on subsystem B
        kp = tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(r1),tf.linalg.LinearOperatorFullMatrix(r2)])
        # print(kp.to_dense())
        part += tf.math.scalar_mul(tf.cast(pr[i],dtype=tf.complex64), kp.to_dense()) # add p[i] r1[i] r2[i]

    return part

# @tf.function
def make_density3q(ypred):
    # make the probability vector and normalize it
    pr = [ypred[i, 0] for i in range(num_pure)]
    pr = tf.nn.softmax(pr)
    # make empty matrix where the CSS will be made
    part = tf.zeros_like(target)

    for i in range(num_pure):  # for each vector in pure state decomp
        m = ypred[i, 1:13]
        m = tf.reshape(m, shape=[3, 2, 2])
        r1 = tf.reshape(tf.complex(m[0, :, 0], m[0, :, 1]), shape=[2, 1])
        r2 = tf.reshape(tf.complex(m[1, :, 0], m[1, :, 1]), shape=[2, 1])
        r3 = tf.reshape(tf.complex(m[2, :, 0], m[2, :, 1]), shape=[2, 1])
        r1 = tf.matmul(r1, r1, adjoint_b=True)  # pure state on subsytem A
        r1 = tf.scalar_mul(1. / tf.linalg.trace(r1), r1)
        r2 = tf.matmul(r2, r2, adjoint_b=True)
        r2 = tf.scalar_mul(1. / tf.linalg.trace(r2), r2)  # pure state on subsystem B
        r3 = tf.matmul(r3, r3, adjoint_b=True)
        r3 = tf.scalar_mul(1. / tf.linalg.trace(r3), r3)  # pure state on subsystem B
        kp = tf.linalg.LinearOperatorKronecker(
            [tf.linalg.LinearOperatorFullMatrix(r1), tf.linalg.LinearOperatorFullMatrix(r2), tf.linalg.LinearOperatorFullMatrix(r3)])
        # print(kp.to_dense())
        part += tf.math.scalar_mul(tf.cast(pr[i], dtype=tf.complex64), kp.to_dense())  # add p[i] r1[i] r2[i]

    return part


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



def custom_loss(y_true, y_pred):
    rho_pred = make_density3q(y_pred)
    return metric_hsd(target, rho_pred)

def generate_test_train_XY(state,num_pure):
    # count=0
    while True:
        x = tf.one_hot(tf.constant(range(num_pure)), depth=num_pure)
        y = np.array([tf.reshape(target,[-1]) for _ in range(num_pure)])
        # count+=1
        yield tf.math.real(x),tf.math.real(y)

def CC_mindelta():
    """stop the training at the end of an epoch if the loss didn't decrease enough"""
    return tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001, patience=1, verbose=1, mode='auto',baseline=None, restore_best_weights=True)


if __name__ == '__main__':
    print(tf.__version__)
    # Build separable approximation of target state:
    global target
    # 3 qubit W state
    target = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1 / 3., 1 / 3., 0, 1 / 3., 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]], dtype='complex64')
    # Bell state 2 qubit
    # target = tf.constant([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype='complex64')
    # print(target)
    global num_pure
    num_pure = 64
    inputs = tf.one_hot(tf.constant(range(num_pure)),depth=num_pure)
    # print(inputs)

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(num_pure,)),
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
    model.compile(optimizer='sgd',loss = custom_loss,metrics=[custom_loss])
    # xy = [data for data in generate_test_train_XY(target, num_pure)]
    # print(tf.shape(xy))
    batchsize =num_pure
    xtrain = inputs
    ytrain = np.array([tf.reshape(target,[-1]) for _ in range(num_pure)])
    print(xtrain, ytrain)
    history = model.fit(
        generate_test_train_XY(target,num_pure),
        batch_size=num_pure,
        epochs=num_pure,
        steps_per_epoch=num_pure**2,
        # epochs=3,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data= generate_test_train_XY(target,num_pure),
        # callbacks = [CC_mindelta()],
        validation_steps = 32, class_weight = None,
        shuffle = False, initial_epoch = 0
    )
    yres = model.predict(inputs)
    print("This is the result of the NN:")
    print(make_density3q(yres))
    verify_density_matrix(make_density3q(yres).numpy())





    # print(inputs)