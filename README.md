# LeNet + MNIST for TensorFlow, Theano and MXNet

## TensorFlow

### Install and configure python packages


```
$ pip install --user keras
$ pip install --user tensorflow
```

In ``~/.keras/keras.json``, change ``"backend"`` value to ``"tensorflow"``

### Run program
```
$ python keras_conv.py 
```

## Theano

### Install and configure python packages

```
$ pip install --user keras
$ pip install --user theano
```

In ``~/.keras/keras.json``, change ``"backend"`` value to ``"theano"``

### Run program
```
$ python keras_conv.py 
```

## MXNet

### Install and configure python packages

```
$ pip install --user mxnet
```

### Run program
```
$ python mxnet_conv.py 
```
