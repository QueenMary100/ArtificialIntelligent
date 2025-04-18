import numpy as np
import matplotlib.pyplot as plt

# Define the function for calculating gradients.
def generate_gradient(X, theta, y):
    sample_count = X.shape[0]
    # Gradient formula: 1/m * X.T * (X*theta - y)
    return (1. / sample_count) * X.T.dot(X.dot(theta) - y)

# Function to read the dataset and extract features + target
def get_training_data(file_path):
    orig_data = np.loadtxt(file_path, skiprows=1)  # Skip header row
    cols = orig_data.shape[1]
    return (orig_data, orig_data[:, :cols - 1], orig_data[:, cols - 1:])

# Initialize the theta parameters
def init_theta(feature_count):
    return np.ones((feature_count, 1))

# Implement gradient descent
def gradient_descending(X, y, theta, alpha):
    Jthetas = []  # Store the loss values to track convergence
    Jtheta = (X.dot(theta) - y).T.dot(X.dot(theta) - y)
    index = 0
    gradient = generate_gradient(X, theta, y)

    while not np.all(np.absolute(gradient) <= 1e-5):  # stop when gradient is close to 0
        theta = theta - alpha * gradient
        gradient = generate_gradient(X, theta, y)
        Jtheta = (X.dot(theta) - y).T.dot(X.dot(theta) - y)

        if (index + 1) % 10 == 0:
            Jthetas.append((index, Jtheta[0]))  # Record every 10 steps

        index += 1

    return theta, Jthetas

# Plot the cost function J(theta) over training steps
def showJTheta(diff_value):
    p_x = []
    p_y = []
    for (index, sum_val) in diff_value:
        p_x.append(index)
        p_y.append(sum_val)
    plt.plot(p_x, p_y, color='b')
    plt.xlabel('steps')
    plt.ylabel('loss function')
    plt.title('Step vs. Loss Function Curve')
    plt.show()

# Plot the original data and the learned linear regression line
def show_linear_curve(theta, sample_training_set):
    x = sample_training_set[:, 1]
    y = sample_training_set[:, 2]
    z = theta[0] + theta[1] * x
    plt.scatter(x, y, color='b', marker='x', label="Sample Data")
    plt.plot(x, z, color="r", label="Regression Curve")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Curve')
    plt.legend()
    plt.show()

# MAIN EXECUTION

# Read dataset
training_data_include_y, training_x, y = get_training_data("./ML/02/lr2_data.txt")

# Get number of samples and features
sample_count, feature_count = training_x.shape

# Define learning rate alpha
alpha = 0.01

# Initialize theta parameters
theta = init_theta(feature_count)

# Run gradient descent to optimize theta
result_theta, Jthetas = gradient_descending(training_x, y, theta, alpha)

# Display final theta values
print("w: {}".format(result_theta[0][0]), "b: {}".format(result_theta[1][0]))

# Visualize loss function curve
showJTheta(Jthetas)

# Visualize regression result
show_linear_curve(result_theta, training_data_include_y)
