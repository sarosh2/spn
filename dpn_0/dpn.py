import networkx as nx
import matplotlib.pyplot as plt
import cupy as cp
import time
import math

class Neuron:
    
    alpha = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    def __init__ (self, activation, input_size):
        self.activation = activation
        self.input_size = input_size
        self.output_size = 0
        self.bias = (float)(cp.random.uniform(0.0, 1.0))
        self.children = []
        self.parents = []
        self.output = 0.0
        self.inputs = None
        self.gradients = None

    def initialize_weights(self):
        
        if self.input_size != 0:
            #He initialization for weights
            std_dev = math.sqrt(2 / self.input_size)
            self.weights = cp.random.randn(1, self.input_size) * std_dev

            #parameters for adam optimizer updates
            self.m_w = cp.zeros_like(self.weights)  # First moment vector
            self.m_b = 0.0
            self.v_w = cp.zeros_like(self.weights)  # Second moment vector
            self.v_b = 0.0

    def add_parent(self, parent):
        self.parents.append(parent)
        self.input_size += 1

    def add_child(self, child):
        self.children.append(child)
        self.output_size += 1
        child.add_parent(self)

    def remove_parent(self, parent):
        if parent in self.parents:
            self.parents.remove(parent)
            self.input_size -= 1

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
            self.output_size -= 1

    def forward_prop(self, input, order):

        if self.inputs is None:
            self.inputs = input
            
        else:
            #append the given input to the list of inputs
            #print(self.inputs.shape, input.shape)
            if order:
                self.inputs = cp.vstack([self.inputs, input])
            else:
                self.inputs = cp.vstack([input, self.inputs])
        
        #if the inputs are complete (all parents have provided an output) propagate the node to its children
        if self.inputs.shape[0] == self.input_size:
            
            #calculate output
            self.output = cp.dot(self.weights, cp.array(self.inputs)) + self.bias
            #APPLY ACTIVATION
            if self.activation == 'Relu':
                self.output = cp.maximum(0, self.output)


            self.temp_inputs = self.inputs
            self.inputs = None
            for child in self.children:
                child.forward_prop(self.output, True)

    def back_prop(self, gradient, t):

        if self.gradients is None:
            self.gradients = gradient
            
        else:
            self.gradients = cp.vstack([gradient, self.gradients]) #append the given input to the list of inputs

        if self.gradients.shape[0] == self.output_size:

            self.gradients = cp.sum(self.gradients, axis=0, keepdims=True)
   
            num_samples = self.gradients.shape[1]
            
            d_output = None
            
            if self.activation == 'Relu':
                d_output = self.gradients * (self.output > 0)

            else:
                d_output = self.gradients


            dW = cp.dot(d_output, self.temp_inputs.T) / num_samples
            db = cp.sum(d_output, axis=1, keepdims=True) / num_samples

            #print(self.weights)
            #print(len(self.parents))
            d_output = cp.dot(self.weights.T, d_output)

            self.m_w = Neuron.beta_1 * self.m_w + (1 - Neuron.beta_1) * dW
            self.m_b = Neuron.beta_1 * self.m_b + (1 - Neuron.beta_1) * db

            # Update biased second moment estimate for weights and biases
            self.v_w = Neuron.beta_2 * self.v_w + (1 - Neuron.beta_2) * cp.square(dW)
            self.v_b = Neuron.beta_2 * self.v_b + (1 - Neuron.beta_2) * cp.square(db)

            # Bias correction for first and second moment estimates
            m_w_hat = self.m_w / (1 - Neuron.beta_1**t)
            m_b_hat = self.m_b / (1 - Neuron.beta_1**t)
            v_w_hat = self.v_w / (1 - Neuron.beta_2**t)
            v_b_hat = self.v_b / (1 - Neuron.beta_2**t)

            # Update the weights and biases
            self.weights -= Neuron.alpha * m_w_hat / (cp.sqrt(v_w_hat) + Neuron.epsilon)
            self.bias -= Neuron.alpha * m_b_hat / (cp.sqrt(v_b_hat) + Neuron.epsilon)

            self.temp_inputs = None
            self.gradients = None

            for i in range(len(self.parents) - 1, -1, -1):
                self.parents[i].back_prop(d_output[-(len(self.parents) - i), :].reshape(1, num_samples), t)

class DPN:

    def __init__ (self):
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

        self.graph = nx.DiGraph()
        self.max_id = 1
        self.vertices = {}
        self.active_node = 0

    def create_node(self, activation, input_size, status):
        neuron = Neuron(activation, input_size)
        
        #add to vertices dict and activate node
        self.vertices[self.max_id] = neuron
        self.active_node = self.max_id
    
        self.graph.add_node(self.active_node)

        if status == 'input':
            self.input_nodes.append(self.active_node)
        elif status == 'output':
            self.output_nodes.append(self.active_node)
            self.vertices[self.active_node].output_size = 1
        else:
            self.hidden_nodes.append(self.active_node)    

        self.max_id += 1

        return self.active_node

    
    def add_connection(self, parent, child):
        self.graph.add_edge(parent, child)
        self.vertices[parent].add_child(self.vertices[child])

    def visualize(self):
        pos = {}

        # Assign positions to input nodes
        for i, node in enumerate(self.input_nodes):
            pos[node] = (1, i)  # Y=3 for input nodes

        # Assign positions to hidden nodes
        for i, node in enumerate(self.hidden_nodes):
            pos[node] = (2, i)  # Y=2 for hidden nodes

        # Assign positions to output nodes
        for i, node in enumerate(self.output_nodes):
            pos[node] = (3, i)  # Y=1 for output nodes

        # Draw the nodes with different colors for input, hidden, and output layers
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.input_nodes, node_color='lightblue', node_size=100, label='Input Layer')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.hidden_nodes, node_color='lightgreen', node_size=100, label='Hidden Layer')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.output_nodes, node_color='lightcoral', node_size=100, label='Output Layer')

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edgelist= self.graph.edges, width=2)

        # Draw labels for the nodes
        nx.draw_networkx_labels(self.graph, pos, font_size=8)

        # Title and display options
        plt.axis('off')  # Hide the axes
        plt.show()

    def compile(self):
        for vertex in self.vertices.values():
            vertex.initialize_weights()
            #print(vertex.input_size)
            #print(vertex.weights.shape)

    def categorical_crossentropy(output, true_labels):
        epsilon = 1e-12
        Y_pred = cp.clip(output, epsilon, 1. - epsilon)

        # Step 2: Compute the log of predicted probabilities
        log_Y_pred = cp.log(Y_pred)

        # Step 3: Compute element-wise multiplication between Y_true and log_Y_pred
        return -cp.sum(true_labels * log_Y_pred, axis=0)

    def caclulate_accuracy(output, true_labels):
        predicted_classes = cp.argmax(output, axis=0)

        # Step 2: Convert one-hot encoded true labels to class indices (index of 1 for each column)
        true_classes = cp.argmax(true_labels, axis=0)

        # Step 3: Compare predicted classes with true classes
        correct_predictions = cp.sum(predicted_classes == true_classes)

        # Step 4: Compute accuracy
        return correct_predictions / true_labels.shape[1]
    
    def softmax(x):
        e_x = cp.exp(x - cp.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
        return e_x / cp.sum(e_x, axis=0, keepdims=True)
    
    def get_batches(X, Y, batch_size):
        num_samples = X.shape[1]
        """
        Generator that yields batches from input matrix X and labels Y.
        X: Input data of shape (n_features, n_samples)
        Y: Labels of shape (n_classes, n_samples)
        batch_size: Number of samples per batch
        """
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            yield X[:, start_idx:end_idx], Y[:, start_idx:end_idx]  # Yield batches of X and Y
    
    def train(self, X, Y, t):
    
        start = time.time()

        for node in self.input_nodes:
            self.vertices[node].forward_prop(cp.array(X), False)

        output = self.vertices[self.output_nodes[0]].output
        for j in range(1, 10):
            output = cp.vstack([output, self.vertices[self.output_nodes[j]].output])

        output = DPN.softmax(output)
        gradient = output - Y

        #back prop
        for i in range(len(self.output_nodes) - 1, -1, -1):
            self.vertices[self.output_nodes[i]].back_prop(cp.array(gradient[i, :].reshape(1, Y.shape[1])), t)

        end = time.time()
        
        #calculate loss
        loss = DPN.categorical_crossentropy(output, Y)
        #print("Loss: ", cp.mean(loss))

        #calculate accuracy
        accuracy = DPN.caclulate_accuracy(output, Y)
        #print("Accuracy: ", cp.mean(accuracy))

        return cp.hstack((end - start, cp.mean(loss), cp.mean(accuracy)))
    
    def execute(self, epochs, batch_size, x_train, y_train, x_val, y_val, x_test, y_test):

        train_metrics = []
        val_metrics = []
        t = 1  # Timestep

        for i in range(epochs):

            metrics = []
            for batch_num, (batch_X, batch_Y) in enumerate(DPN.get_batches(x_train, y_train, batch_size)):
                metrics.append(self.train(batch_X, batch_Y, t))
                t += 1

            metrics = cp.vstack(metrics)
            
            #validate output
            for node in self.input_nodes:
                self.vertices[node].forward_prop(cp.array(x_val), False)

            output = self.vertices[self.output_nodes[0]].output

            for j in range(1, 10):
                output = cp.vstack([output, self.vertices[self.output_nodes[j]].output])
            val_loss = cp.mean(DPN.categorical_crossentropy(output, y_val))
            val_accuracy = cp.mean(DPN.caclulate_accuracy(output, y_val))

            print(f"Epoch: {i + 1} Total_Time: {cp.sum(metrics[:, 0]):.4f} Average_Time_per_batch: {cp.mean(metrics[:, 0]):.4f} Train_Accuracy: {metrics[-1, 2]:.4f} Train_Loss: {metrics[-1, 1]:.4f} Val_Accuracy: {val_accuracy:.4f} Val_Loss: {val_loss:.4f}")
            train_metrics.append(metrics)
            val_metrics.append([val_loss, val_accuracy])

        for node in self.input_nodes:
            self.vertices[node].forward_prop(cp.array(x_test), False)

        output = self.vertices[self.output_nodes[0]].output
        for j in range(1, 10):
            output = cp.vstack([output, self.vertices[self.output_nodes[j]].output])

        #print(output.shape, x_test_flat.shape)
        test_loss = cp.mean(DPN.categorical_crossentropy(output, y_test))
        test_accuracy = cp.mean(DPN.caclulate_accuracy(output, y_test))

        print("Test_Accuracy: ", test_accuracy, "Test_Loss: ", test_loss)

        return train_metrics, val_metrics, [test_loss, test_accuracy]