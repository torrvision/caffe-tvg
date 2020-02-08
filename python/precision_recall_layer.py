# Need to have the correct Caffe in your "PYTHONPATH!!"
import caffe

import numpy as np
import sklearn.metrics as metrics

class PrecisionRecall(caffe.Layer):
    """ bottom[0] is the prediction
        bottom[1] is a binary vector of the ground truth
    """

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Should have exactly two bottoms")

        self.i = 0
        self.options = eval(self.param_str)

        if 'test_iterations' not in self.options:
            raise Exception("The parameter string should be a dictionary with the 'test_iterations' key")
        self.test_iterations = self.options['test_iterations']

        self.initialised = False


    def reshape(self, bottom, top):

        n,c = bottom[0].shape[0], bottom[0].shape[1]
        self.n = n
        self.c = c

        top[0].reshape(1)
        top[1].reshape(1,c)

        #if h != 1 or w != 1:
        #    raise Exception("h = {} and w = {}. Should both be 1".format(h, w))
    
        if not self.initialised:
            self.labels = np.zeros((n*self.test_iterations, c))
            self.scores = np.zeros((n*self.test_iterations, c))
            self.initialised = True


    def forward(self, bottom, top):

        n = self.n
        scores_batch = bottom[0].data
        scores_batch = np.squeeze(scores_batch)

        labels_batch = bottom[1].data
        labels_batch = np.squeeze(labels_batch)

        self.scores[self.i : (self.i + 1)*n,:] = scores_batch
        self.labels[self.i : (self.i + 1)*n,:] = labels_batch
        
        top[0].data[...] = 0
        top[1].data[...] = np.zeros((self.c))
        
        self.i += 1

        if self.i == self.test_iterations:
            # Compute the answer
            ap_per_class = metrics.average_precision_score(self.labels, self.scores, average=None)

            top[1].data[...] = ap_per_class * self.test_iterations # This is for display purposes. Since Caffe averages the results when it prints
            top[0].data[...] = np.mean(ap_per_class) * self.test_iterations

            self.labels *= 0
            self.scores *= 0
            self.i = 0
        

    def backward(self, bottom, top):
        raise Exception("This is a test-time layer. Backward should not be called")
