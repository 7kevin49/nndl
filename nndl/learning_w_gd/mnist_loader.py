import numpy as np
from sklearn.datasets import fetch_openml

def load_data():
    """Return the MNIST data as a tuple containing the training, validation,
    and test data. Each is a tuple (images, labels)."""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    # Convert data to float32 and targets to integers
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)
    
    # Split the data into training (first 50k), validation (next 10k), and test (last 10k)
    training_data = (X[:50000], y[:50000])
    validation_data = (X[50000:60000], y[50000:60000])
    test_data = (X[60000:], y[60000:])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    """Return (training_data, validation_data, test_data) in a format convenient for neural network code."""
    tr_d, va_d, te_d = load_data()
    
    # Process training data: for each image x and label y, reshape x to (784,1) and vectorize the label.
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # Process validation and test data: reshape images and leave labels as integers.
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)
