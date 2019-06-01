import numpy
import math

class NeuralNetwork:
    # Accepts an array of desired layer sizes. Ex. Network([16,16,16,10]) would
    # Create a 16 input, 10 output neural network with two, 16 neuron hidden
    # Layers. This setup makes it really easy to experiment with different combinations
    def __init__(self, layers):
        # List of numpy arrays representing each layer of the neuron
        self.network = []
        # List of numpy arrays representing each set of weights in the network
        self.weights = []
        # List of numpy arrays representing the biases for each layers
        self.biases = []

        # iterate through the input, populating the network and weights.
        for i,layer in enumerate(layers[:-1]):
            self.network.append(numpy.zeros((layer,1)))
            # Random values for the weights, between -1/sqrt(n) and 1/sqrt(n), where n is the number of inputs
            self.weights.append((numpy.random.random((layers[i+1],layer))*.2-.1))#(2/math.sqrt(layers[0]))-1/math.sqrt(layers[0]))
            # Small initial bias for each neuron
            self.biases.append(numpy.full((layers[i+1],1), .5))

        # Fill the last layer with zeros. Not necessary, but I don't like null
        # values
        self.network.append(numpy.zeros((layers[-1],1)))

    # simple sigmoid, applied to a 1D matrix
    def sigmoid(self,x):
        return 1/(1+numpy.exp(-x))

    # Runs an input through the network and returns the output
    def forwardProp(self,inputs):
        # Swap first layer with the input
        inputs = numpy.array(inputs)
        inputs.shape = (self.network[0].size,1)
        self.network[0]=inputs

        # For each layer, run sum the weights and biases, then apply the
        # Sigmoid function
        for i,layer in enumerate(self.network[:-1]):
            # Before Sigmoid
            raw_output = self.weights[i].dot(self.network[i])#+self.biases[i]
            # Apply result to next layer
            self.network[i+1] = self.sigmoid(raw_output)

        # Return output
        return self.network[-1]

    def getOutput(self):
        max_value = max(list(self.network[-1]))
        return list(self.network[-1]).index(max_value)

    # Derivative of the sigmoid function
    def dSigmoid(self, output):
	       return self.sigmoid(output) * (1.0 - self.sigmoid(output))

    # Simple function for calculating the error in the output
    def error(self, target):
        return numpy.subtract(target,self.network[-1])#*self.dSigmoid(self.network[-1])

    # The back propagation function, for adjusting the weights of the neurons
    def backProp(self, target):
        # Calculate the error of the output
        target.shape=(target.size,1)
        outputError = self.error(target)
        # This will hold the calculated neuron changes
        self.deltas = []
        self.biasDeltas = []
        errorArray = []
        # Set up the delta matrix
        for i in range(len(self.network)):
            self.deltas.append([])
            self.biasDeltas.append([])
            errorArray.append([])

        # Iterate the layers backwards and perform the propogation on each neuron
        for i,layer in reversed(list(enumerate(self.network))):
            # For holding the total error from the neurons in the next layer up

            # Populate the errors in the last layer first
            if i == len(self.network)-1:
                errorArray[i]=outputError
                #for j,neuron in enumerate(layer):
                #    errors.append(target[j] - neuron)

            # If not last layer, do normal backpropogation
            else:
                # For each neuron in the current layer
                errors = []
                for j in range(len(layer)):
                    # Holds total error so far
                    temperror = 0
                    # For each neuron in the layer above
                    for c,neuron in enumerate(self.network[i+1]):
                        # Calculate the error from that neuron and add it to total
                        temperror += self.weights[i][c][j] * self.deltas[i+1][c]
                    errors.append(temperror)
                errorArray[i]=errors

                biaserror = 0
                for c,neuron in enumerate(self.network[i+1]):
                    biaserror += self.biases[i][c]*self.deltas[i+1][c]

                self.biasDeltas[i] = biaserror*self.dSigmoid(.5)


            # Add the error for this particular neuron to the error array
            for j,neuron in enumerate(layer):
                self.deltas[i].append(errorArray[i][j] * self.dSigmoid(neuron))
            self.deltas[i] = numpy.array(self.deltas[i])
        self.deltas = numpy.array(self.deltas)

    def updateWeights(self, learningRate):
        for i,layer in enumerate(self.network):
            if i!=0:
                for j,neuron in enumerate(layer):
                    for c,input in enumerate(self.network[i-1]):
                        self.weights[i-1][j][c]+=learningRate*numpy.array(self.deltas[i][j])*input
                    #self.biases[i-1][j]=self.biases[i-1][j]+self.biasDeltas[i-1]*learningRate#*input(self.biases[i])


    def train(self,dataSet,target,trainingSpeed):
        for i,data in enumerate(dataSet):
            self.forwardProp(data)
            self.backProp(numpy.array(target[i]))
            self.updateWeights(trainingSpeed)


    # Returns the error between the desired output and the actual output
    def getError(self, target):
        target = numpy.array(target)
        target.shape = (len(target),1)
        meanDiff = numpy.mean(numpy.absolute(target - self.network[-1]))
        return meanDiff
