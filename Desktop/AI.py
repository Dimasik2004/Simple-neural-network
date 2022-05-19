import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



training_inputs = np.array ( [[0,1,0,0],[0,0,1,0],[0,1,1,1],[1,1,1,0]
])



training_outputs=np.array( [[0,0,1,1]]).T



np.random.seed (1)
print(training_inputs.shape[1])
synaptic_weights = 2 * np.random.random ((training_inputs.shape[1],1)) - 1
print("random weights")
print(synaptic_weights)


#learn
for i in range (20000):
    input_layer= training_inputs
    outputs=sigmoid( np.dot (input_layer, synaptic_weights))
    err = training_outputs - outputs
    amendments = np.dot( input_layer.T, err * (outputs * (1-outputs)))
    synaptic_weights += amendments

print( "weights after learning" )

print(synaptic_weights)

print( "Result after learning" )
print(outputs)