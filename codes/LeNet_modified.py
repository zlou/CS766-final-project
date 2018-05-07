import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle

def create_network(input,is_training,scope='LeNet',reuse=False):

    """setting up parameters"""
    num_maps={
        'layer_1': 30,
        'layer_2': 35,
        'layer_fully_1': 220,
        'layer_fully_2': 100,
        'layer_fully_3': 25
    }
    conv_kernel_height=3
    conv_kernel_width=3
    pool_kernel_height=2
    pool_kernel_width=2

    """creating network"""
    with tf.variable_scope(scope,reuse=reuse):

        with slim.arg_scope([slim.conv2d],padding='VALID',activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training,'updates_collections':None}):

            net=slim.conv2d(input,6,[conv_kernel_height,conv_kernel_width],scope='conv1')
            net=slim.max_pool2d(net,[pool_kernel_height,pool_kernel_width],scope='pool1')
            net=slim.conv2d(net,num_maps['layer_2'],[conv_kernel_height,conv_kernel_width],scope='conv2')
            net=slim.max_pool2d(net,[pool_kernel_height,pool_kernel_width],scope='pool2')
            net=slim.flatten(net,scope='flatten')  # flatten layer
            net=slim.fully_connected(net,num_maps['layer_fully_1'],activation_fn=tf.nn.relu,scope='fully_connected1')
            net=slim.fully_connected(net,num_maps['layer_fully_2'],activation_fn=tf.nn.relu,scope='fully_connected2')
            net=slim.fully_connected(net,num_maps['layer_fully_3'])
        return net
def evaluate(x_data,y_data,batch_size,num_channel=25):
    n_examples=len(x_data)
    x = tf.placeholder(tf.float32, (None, 15, 15, num_channel))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 25)
    net=create_network(x,False,scope='lenet',reuse=True)
    correct_prediction=tf.equal(tf.argmax(net,1),tf.argmax(one_hot_y,1))
    evaluate_operation=tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
    total_accuracy=0
    num_batch=int(np.ceil(len(x_data)/batch_size))
    sess=tf.get_default_session()
    for batch_index in range(num_batch):
        x_batch,y_batch=x_data[batch_index*batch_size:(batch_index+1)*batch_size],y_data[batch_index*batch_size:(batch_index+1)*batch_size]
        accuracy=sess.run(evaluate_operation,feed_dict={x:x_batch,y:y_batch})
        total_accuracy+=accuracy*len(x_batch)
       # print('x_batch',len(x_batch),'  batch_size',batch_size)

    return total_accuracy/n_examples
def train(x_train,y_train,x_validation,y_validation,num_channel=25,b_size=100,l_rate=0.001,n_epoch=100): # input is one-hot encoded
    """ Define the graph"""

    x = tf.placeholder(tf.float32, (None, 15, 15, num_channel))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 25)
    net=create_network(x,True,scope='lenet',reuse=False) # create network
    # compute cross-entropy loss
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=net,labels=one_hot_y) # getting the vector of softmax cross entropy loss (not averaged yet)
    loss=tf.reduce_mean(cross_entropy)
    # setting up the optimizer to use
    optimizer=tf.train.AdamOptimizer(learning_rate=l_rate)
    training_operation=optimizer.minimize(loss)
    # running the graph
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_training_eg=len(x_train)

        num_epoch=n_epoch
        batch_size=b_size
        num_batch=int(np.ceil(num_training_eg/batch_size))
        print('number of batches',num_batch)
        print('training')
        print()
        for epoch in range(num_epoch):
            x_train,y_train=shuffle(x_train,y_train)
            for batch_index in range(num_batch):
                x_batch,y_batch=x_train[batch_index*batch_size:(batch_index+1)*batch_size],y_train[batch_index*batch_size:(batch_index+1)*batch_size]
                sess.run(training_operation,feed_dict={x:x_batch,y:y_batch})
            validation_accuracy=evaluate(x_validation,y_validation,b_size)
            print(validation_accuracy)

        saver.save(sess, 'E:/CS 766 computer vision/UW-Madison\project/research\LeNet for choosing number of windows/lenet_modified.ckpt')
        # print("Model saved")

def test(test_x,test_y,batch_size):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'E:/CS 766 computer vision/UW-Madison\project/research\LeNet for choosing number of windows/lenet_modified.ckpt')

        test_accuracy = evaluate(test_x, test_y,batch_size)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
def predict(x_data,num_channel=25,batch_size=1):
    x = tf.placeholder(tf.float32, (None, 15, 15, num_channel))

    net = create_network(x, True, scope='lenet', reuse=False)  # create network
    prediction_operation=tf.argmax(net,1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # tf.train.get_checkpoint_state‌​()
        # states = tf.train.get_checkpoint_state‌​('E:/CS 766 computer vision/UW-Madison/project/research/LeNet for choosing number of windows'‌​)
        # checkpoint_paths = states.all_model_checkpoi‌​nt_paths
        #saver.recover_last_checkpoints(checkpoint_paths:{'E:/CS 766 computer vision/UW-Madison/project/research/LeNet_for_choosing_number_of_windows'}‌​)
        saver.restore(sess,'E:/CS 766 computer vision/UW-Madison\project/research\LeNet for choosing number of windows/lenet_modified.ckpt')
        #saver.restore(sess,'E:/CS 766 computer vision/UW-Madison\project/research\LeNet for choosing number of windows/lenet_modified')
        x_batch= x_data[0:batch_size]
        prediction = sess.run(prediction_operation, feed_dict={x: x_batch})
    return prediction
