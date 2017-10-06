import copy, numpy as np
np.random.seed(0)

#import file

data = []
totalrow = 0

with open("C:\\Users\\insyi\\Desktop\\RNN\\JA004860.csv") as f:
    next(f)
    for count,i in enumerate(f):
        content = i.split(",")
        content[-1] = content[-1].strip()
        data.append(content)
        totalrow = totalrow + 1

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# input variables
learningrate = 0.001
input_dimension = 16
hidden_dimension = 48
output_dimension = 1
epochs = 1

# initialize neural network weights
synapse_0 = 2*np.random.random((input_dimension,hidden_dimension)) - 1
synapse_1 = 2*np.random.random((hidden_dimension,output_dimension)) - 1
synapse_h = 2*np.random.random((hidden_dimension,hidden_dimension)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(epochs):
    for ii in range(totalrow):
        
        example = data[ii]

        #convert values to float
        a=[]
        b=[]

        for i in range(19):
            #input for training data
            if i >= 2 and i<=17:
                a.append(float(example[i]))
            #answer for training data
            if i==18:
                b.append(float(example[i]))

        # where we'll store our best guess (binary encoded)
        c = np.zeros_like(b)

        overallError = 0

        output_derivative = list()
        hidden_values = list()
        hidden_values.append(np.zeros(hidden_dimension))


        #input and output
        X = np.array([a])
        y = np.array([b])


        # hidden layer (input ~+ prev_hidden)
        hidden_layer = sigmoid(np.dot(X,synapse_0) + np.dot(hidden_values[-1],synapse_h))

        # output layer 
        output_layer = sigmoid(np.dot(hidden_layer,synapse_1))

        # did we miss?... if so, by how much?
        output_error = y - output_layer
        output_derivative.append((output_error)*sigmoid_output_to_derivative(output_layer))
        overallError += np.abs(output_error[0])

        # decode estimate so we can print it out
        d = output_layer

        # store hidden layer so we can use it in the next timestep
        hidden_values.append(copy.deepcopy(hidden_layer))


        future_hidden_derivative = np.zeros(hidden_dimension)
        
        
        # start backpropagating the derivatives
        
        # selecting the current hidden layer from the list.
        hidden_layer = hidden_values[-1]
        # selecting the current hidden layer from the list.
        prev_hidden_layer = hidden_values[-2]


        # error at output layer
        output_derivative = output_derivative[-1]
        # error at hidden layer
        hidden_derivative = (future_hidden_derivative.dot(synapse_h.T) + output_derivative.dot(synapse_1.T)) * sigmoid_output_to_derivative(hidden_layer)
        
        #end of backpropagation
        
        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(hidden_layer).T.dot(output_derivative)
        synapse_h_update += np.atleast_2d(prev_hidden_layer).T.dot(hidden_derivative)
        synapse_0_update += X.T.dot(hidden_derivative)

        future_hidden_derivative = hidden_derivative

        synapse_0 += synapse_0_update * learningrate
        synapse_1 += synapse_1_update * learningrate
        synapse_h += synapse_h_update * learningrate    

        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

    print("Epochs : #" + str(j+1))
    print ("Pred:" + str(d))
    print ("True:" + str(b))
    print ("Error:" + str(overallError))
    print ("----------------------")