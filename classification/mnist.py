from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow_quantum as tfq
import tensorflow as tf

class MNISTData:
    def __init__(self, seed):
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        self.data = mnist['data']
        self.labels = np.array(mnist['target'], dtype=np.int8)
        self.seed = seed
        
    def _get_binary_data_encoding(self):
        labels_zero = self.labels[self.labels==0] + 1
        labels_one = self.labels[self.labels==1] - 2
        binary_labels = np.hstack((labels_zero, labels_one))
        digits_zero = self.data[self.labels==0]
        digits_one = self.data[self.labels==1]
        binary_digits = np.vstack((digits_zero, digits_one))
        
        pca = PCA(n_components=8)
        sc = StandardScaler()
        binary_digits = sc.fit_transform(binary_digits)
        data = pca.fit_transform(binary_digits)
        data = np.expand_dims(data, axis=2)
        data = (data-np.min(data))/(np.max(data)-np.min(data))
        data = np.concatenate((np.cos(np.pi*0.5*data), np.sin(np.pi*0.5*data)), axis=2)
        
        qubits = cirq.GridQubit.rect(1,8)
        _data = []
        for idx in range(data.shape[0]):
            datapoint = data[idx]
            _circuit = cirq.Circuit()
            for i in range(8):
                _circuit.append(encode_classical_datapoint(np.array(datapoint[i][::-1]), [qubits[i]]))
            _data.append(_circuit)
        return _data, binary_labels
        
    # returns data in qubit encoding
    def get_binary_test_train_split(self):
        data, labels = self._get_binary_data_encoding()
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=self.seed)
        return tfq.convert_to_tensor(X_train), tfq.convert_to_tensor(X_test), y_train, y_test
    
    # returns tensorflow objects of raw mnist data
    def get_three_five_test_train_split(self):
        labels_three = self.labels[self.labels==3] - 2
        labels_five = self.labels[self.labels==5] - 6
        
        labels = np.hstack((labels_three, labels_five))
        digits = np.vstack((self.data[self.labels==3], self.data[self.labels==5]))
        X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=0.1, random_state=self.seed)
        return tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test), y_train, y_test
        