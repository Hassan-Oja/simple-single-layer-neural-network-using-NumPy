import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

# Inputs for the OR Gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([0, 1, 1, 1]) 

weights = np.random.rand(2)
bias = np.random.rand()
learning_rate = 0.1

for epoch in range(50):
    print(f"Epoch {epoch + 1}")
    for i in range(len(inputs)):
        input_data = inputs[i]
        expected_output = expected_outputs[i]

        weighted_sum = np.dot(input_data, weights) + bias

        prediction = step_function(weighted_sum)

        error = expected_output - prediction

        weights += learning_rate * error * input_data
        bias += learning_rate * error

        print(f"Input: {input_data}, Expected: {expected_output}, Predicted: {prediction}, Error: {error}")
    print(f"Updated Weights: {weights}, Bias: {bias}\n")

print("Testing the OR Gate Neural Network")
for i in range(len(inputs)):
    input_data = inputs[i]
    weighted_sum = np.dot(input_data, weights) + bias
    prediction = step_function(weighted_sum)
    print(f"Input: {input_data}, Predicted Output: {prediction}")
