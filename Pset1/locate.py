import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="training options")
parser.add_argument('--src-path', action='store', dest='src_path', required=True)
parser.add_argument('--res-path', action='store', dest='res_path', required=True)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', required=True)
args = parser.parse_args()

filePath = args.src_path
resPath = args.res_path
gpu_num = args.gpu_num
#filePath = './test_case_ans/sample_case002_input.txt'
#resPath = './ans'

c = 3e8

def loadData(filePath):
    with open(filePath, 'r') as fd:
        station_num = int(fd.readline())
        device_num = int(fd.readline())
        dimen = int(fd.readline())
        station_cord = []
        for i in range(station_num):
            cord = fd.readline().split()
            cord = [np.float64(i) for i in cord]
            station_cord.append(cord)
        station_cord = np.array(station_cord)

        device_cord = []
        for i in range(device_num):
            cord = fd.readline().split()
            cord = [np.float64(i) for i in cord]
            device_cord.append(cord)
        device_cord = np.array(device_cord)
        return station_cord, device_cord, dimen

def loadRes(filePath):
    with open(filePath, 'r') as fd:
        device_cord = []
        for line in fd:
            cord = [np.float64(i) for i in line.split()]
            device_cord.append(cord)
    device_cord = np.array(device_cord)
    return device_cord



S, D, dimen = loadData(filePath)
device_Dist = D*c

cord_init = np.mean(S, axis=0)
cord_init = cord_init.reshape([1, dimen]) * np.ones([len(D), dimen])
bias_init = np.zeros(len(D))
# cord_init = loadRes(resPath)

#cut=10
#device_Dist = device_Dist[:10]
#cord_init = cord_init[:10]
#bias_init = bias_init[:10]

with tf.device('/gpu:'+gpu_num):
#     cord_x = tf.Variable(cord_init[:, 0], name='Cordinate_x', trainable=False) #(N,)
#     cord_y = tf.Variable(cord_init[:, 1], name='Cordinate_y', trainable=False)
#     cord_z = tf.Variable(cord_init[:, 2], name='Cordinate_z', trainable=False)
    cord = tf.Variable(cord_init, name='Cordinate') #(N,3)
    Station = tf.constant(S, tf.float64) #(M,3)
    bias = tf.Variable(bias_init, 'Bias') #(N, )

    alpha1 = tf.Variable(np.ones(len(cord_init)), 'Alpha1', dtype = tf.float64)  #(N,)
    alpha2 = tf.Variable(np.ones(len(cord_init)), 'Alpha2', dtype = tf.float64)  #(N,)
    alpha3 = tf.Variable(np.ones(len(cord_init)), 'Alpha3', dtype = tf.float64)  #(N,)
    beta1 = tf.Variable(np.ones(len(S)), 'Beta1', dtype = tf.float64)  #(M,)

    D_dist = tf.constant(device_Dist, tf.float64)
    mask = tf.constant([[[1, 1, 1]]], tf.float64)

    expand_cord = tf.expand_dims(cord, 1) #(N, 1, 3)
    expand_Station = tf.expand_dims(Station, 0) #(1, M, 3)

    expand_alpha1 = tf.expand_dims(alpha1, 1) #(N, 1)
    expand_alpha2 = tf.expand_dims(alpha2, 1)
    expand_alpha3 = tf.expand_dims(alpha3, 1)

    expand_beta1 = tf.expand_dims(beta1, 0)

    expand_bias = tf.expand_dims(bias, 1)

    dist_2 = tf.reduce_sum((expand_cord - expand_Station)**2, 2) #(N, M)

    global_step = tf.Variable(0, name='global_step', trainable=False)


    # dist_hat = tf.sqrt(expand_alpha1**2 * dist_x +expand_alpha2**2 * dist_y +
    #                    expand_alpha3**2 * dist_z) - expand_bias #(N,M)
    # dist_hat = expand_alpha1 * tf.sqrt(dist_x + dist_y + expand_alpha2**2) - expand_bias #(N,M)
    dist_hat = expand_alpha1 * tf.sqrt(dist_2) - expand_bias  #(N,M)


    losses = tf.reduce_sum((dist_hat - D_dist) **2, 1)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.85, staircase=True) + tf.train.exponential_decay(1e-2, global_step, 5000, 0.85, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(losses, global_step=global_step)

config=tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
# with tf.Session(config=config) as sess:
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())
for i in range(50000):
    _, loss = sess.run([train_op, losses])
    if i%1000 == 0:
        print np.sum(loss), 'lr=', sess.run(learning_rate)
B, P, a1 = sess.run([bias, cord, alpha1])

with open(resPath, 'w') as fd:
    if P.shape[1] ==3:
        for i in range(len(P)):
            fd.write('{}\t{}\t{}\n'.format(P[i, 0], P[i, 1], P[i, 2]))
    else:
        for i in range(len(P)):
            fd.write('{}\t{}\n'.format(P[i, 0], P[i, 1]))
