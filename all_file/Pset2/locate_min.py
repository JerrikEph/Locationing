import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="training options")
parser.add_argument('--src-path', action='store', dest='src_path', required=True)
parser.add_argument('--res-path', action='store', dest='res_path', required=True)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', required=True)
parser.add_argument('--num-station', action='store', dest='num_station', required=True)
args = parser.parse_args()
filePath = args.src_path
resPath = args.res_path
num_station = int(args.num_station)
gpu_num = args.gpu_num

c = 3e8

def get_top_k(file_name, save_file=None, top_k=10):

    fin = open(file_name, 'rb')
    base_dict = {}
    top_k_toa = []
    top_k_base = []
    for line_idx, line in enumerate(fin):
        if line_idx == 0:
            N = int(line)
        elif line_idx == 1:
            M = int(line)
        elif line_idx == 2:
            pass
        elif line_idx < 3+N:
            base_dict[line_idx-3] = [float(x) for x in line.split()]
        else:
            toa_list = [(i, float(t)) for i, t in enumerate(line.split())]
            # check
            if len(toa_list) != N:
                print 'wrong line!'
                exit()
            toa_list = sorted(toa_list, lambda x, y: cmp(x[1], y[1]))
            cur_toa = []
            cur_base = []
            for (i, t) in toa_list[:top_k]:
                cur_toa.append(t)
                cur_base.append(base_dict[i])
            top_k_toa.append(cur_toa)
            top_k_base.append(cur_base)
    if save_file is not None:
        cPickle.dump([top_k_toa, top_k_base], open(save_file, 'wb'))
    return np.array(top_k_toa), np.array(top_k_base), 3

D, S, dimen = get_top_k(filePath, top_k=num_station)
device_Dist = D*c

cord_init = np.mean(S, axis=1)
bias_init = np.zeros(len(D))


with tf.device('/gpu:'+gpu_num):

    cord = tf.Variable(cord_init, name='Cordinate') #(N,3)
    Station = tf.constant(S, tf.float64) #(N, M,3)
    bias = tf.Variable(bias_init, 'Bias') #(N, )

    alpha1 = tf.Variable(np.ones(len(cord_init)), 'Alpha1', dtype = tf.float64)  #(N,)
    alpha2 = tf.Variable(np.ones(len(cord_init)), 'Alpha2', dtype = tf.float64)  #(N,)
    alpha3 = tf.Variable(np.ones(len(cord_init)), 'Alpha3', dtype = tf.float64)  #(N,)
    beta1 = tf.Variable(np.ones(len(S)), 'Beta1', dtype = tf.float64)  #(M,)

    D_dist = tf.constant(device_Dist, tf.float64)
    mask = tf.constant([[[1, 1, 1]]], tf.float64)

    expand_cord = tf.expand_dims(cord, 1) #(N, 1, 3)
    expand_Station = Station #(N, M, 3)

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