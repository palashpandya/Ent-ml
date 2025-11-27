import tensorflow as tf

def metric_hsd(r1,r2):
    return tf.linalg.trace(tf.linalg.matmul((r1-r2), (r1-r2), adjoint_a = True))

def make_density(ypred):
    pr=[0.,0.,0.,0.]
    part = tf.zeros(shape=[4,4],dtype='complex64')
    totpr = 0.
    for i in range(4):
        pr[i] = ypred[i, 0]
        totpr += pr[i]
    pr = tf.math.scalar_mul(1/totpr,pr)

    for i in range(4):
        a1r,a1i,b1r,b1i,a2r,a2i,b2r,b2i = ypred[i,1:9]
        r1 = [[tf.complex(a1r , a1i)],[tf.complex(b1r, b1i)]]
        r2 = [[tf.complex(a2r, a2i)], [tf.complex(b2r, b2i)]]
        r1 = tf.matmul(r1,r1,adjoint_b=True)
        r2 = tf.matmul(r2, r2, adjoint_b=True)
        part += tf.math.scalar_mul(pr[i],tf.linalg.LinearOperatorKronecker([r1,r2]))

    return part

def custom_loss(y_true, y_pred):
    y_true = target
    rho_pred = make_density(y_pred)
    return metric_hsd(y_true, rho_pred)


if __name__ == '__main__':
    print(tf.__version__)
    # Build separable approximation of target state:
    global target
    target = tf.constant([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype='complex64')
    # print(target)
    inputs = tf.one_hot(tf.constant([0,1,2,3]),depth=4)
    print(inputs)

    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(None,4)),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(4*9,activation='softmax')
        ]
    )
    result = model.predict(inputs[1])
    print(model.summary())
    print(tf.reduce_sum(result[1]))
    model.compile(optimizer='adadelta',loss=custom_loss, metrics=[metric_hsd])
    print(result)



    # print(inputs)