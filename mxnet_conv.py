from __future__ import print_function

from data_loader import load_data
import mxnet as mx
import time

import matplotlib.pyplot as plt

import logging
logging.getLogger().setLevel(logging.DEBUG)

(train_img, train_lbl), (test_img, test_lbl) = load_data()

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 10

train_iter = mx.io.NDArrayIter(train_img, train_lbl, BATCH_SIZE)
test_iter = mx.io.NDArrayIter(test_img, test_lbl, BATCH_SIZE)

print('Using MXNet backend.')

data = mx.symbol.Variable(name='data')
# CONV1->RELU1->POOL1
conv1 = mx.sym.Convolution(data=data, num_filter=20, kernel=(5,5), stride=(1,1), dilate=(1,1), cudnn_off=True)
relu1 = mx.sym.Activation(data=conv1, act_type='relu')
pool1 = mx.sym.Pooling(data=relu1, pool_type='max', kernel=(2,2), stride=(2,2))
# CONV2->RELU2->POOL2
conv2 = mx.sym.Convolution(data=pool1, num_filter=50, kernel=(5,5), stride=(1,1), dilate=(1,1), cudnn_off=True)
relu2 = mx.sym.Activation(data=conv2, act_type='relu')
pool2 = mx.sym.Pooling(data=relu2, pool_type='max', kernel=(2,2), stride=(2,2))
# FLATTEN->FC1->RELU3
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
relu3 = mx.sym.Activation(data=fc1, act_type='relu')
# FC2->SOFTMAX
fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=NUM_CLASSES)
label = mx.sym.Variable(name='softmax_label')
net = mx.sym.SoftmaxOutput(data=fc2, name='softmax', label=label)

# We visualize the network structure with output size (the BATCH_SIZE is ignored.)
shape = {'data' : (BATCH_SIZE, 1, 28, 28)}
mx.viz.plot_network(symbol=net, shape=shape)

optimizer_params = {
    'learning_rate': 0.1, # default=0.1
    'momentum': 0.0, # default=0.9
    'wd': 0.0, # default=0.0001
    'lr_scheduler': None,
    'clip_gradient': 0.0
}

callbacks = [
    mx.callback.log_train_metric(period=BATCH_SIZE, auto_reset=False),
    mx.callback.Speedometer(batch_size=BATCH_SIZE, frequent=BATCH_SIZE)
]

# Compile
model = mx.module.Module(
    symbol = net,
    context = mx.cpu(0), # use GPU 0 for training, others are same as before
    data_names = ['data'],
    label_names = ['softmax_label'])
# Fit
start_time = time.time()
model.fit(
    train_data=train_iter,
    eval_data=test_iter, # validation after each epoch
    eval_metric=['accuracy'],
    optimizer='sgd',
    optimizer_params=optimizer_params,
    num_epoch=NUM_EPOCHS,
    batch_end_callback=callbacks,
    eval_end_callback=callbacks,
    eval_batch_end_callback=callbacks
)
elapsed_time = time.time() - start_time
print('Fit + Eval time: ', elapsed_time)

######## DEPRECATED - LEFT HERE FOR COMPARISON WITH mxnet.mod.Module() ########
# # Compile
# model = mx.model.FeedForward(
#     symbol = net,
#     ctx = mx.gpu(0), # use GPU 0 for training, others are same as before
#     num_epoch = NUM_EPOCHS,
#     optimizer='sgd',
#     learning_rate=0.1,
#     momentum=0.0, # default=0.9
#     wd=0.0, # default=0.0001
#     lr_scheduler=None,
#     clip_gradient=0.0
# )
# # Fit
# model.fit(
#     X=train_iter,
#     eval_data=test_iter, # validation after each epoch
#     eval_metric=['mse', 'accuracy'],
#     batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 200) # output progress for each 200 data batches
# )
###############################################################################
