import torch
import torch.nn as nn
import torch.autograd as autograd

class DPNAutoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size, total_nodes, biases, *weights):

        input_size = input.shape[1]

        nodes = 0
        for i in range(len(weights)):
            output = input.matmul(weights[i].T) + biases[i]
            if nodes < total_nodes - output_size:  # if the current block has non-output neurons
                output = output.relu()  # apply ReLU activation
            input = torch.cat((input, output), dim=1)
            nodes += weights[i].shape[0]

        ctx.input_size = input_size
        ctx.nodes = total_nodes
        ctx.output_size = output_size
        ctx.save_for_backward(input, *weights)

        return input[:, -output_size:]

    @staticmethod
    def backward(ctx, grad_output):

        #retrieve saved tensors
        saved = ctx.saved_tensors
        input = saved[0]
        weights = saved[1:]

        #variables used for gradient calculation
        output_size = ctx.output_size
        input_size = ctx.input_size
        nodes = ctx.nodes

        batch_size = input.shape[0]

        d_weights = []
        
        d_output = input.new_zeros(input.shape)
        d_output[:, -output_size:] = grad_output

        output_end_idx = input_size + nodes
        for i in reversed(range(len(weights))):

            output_start_idx = output_end_idx - weights[i].shape[0]

            d_o = d_output[:, output_start_idx : output_end_idx]

            if output_start_idx < input_size + nodes - output_size: #if the current block has non-output neurons
                d_o *= (input[:, output_start_idx : output_end_idx] > 0).float() #apply relu gradient

            d_weights.insert(0, d_o.T.matmul(input[:, :output_start_idx]) / batch_size) #add to the beginning of d_weights
            d_output[:,  :output_start_idx] += weights[i].T.matmul(d_o.T).T
            output_end_idx -= weights[i].shape[0]

        d_biases = d_output[:, input_size:].sum(0) / batch_size
        d_input = d_output[:, :input_size]

        return d_input, None, None, d_biases, *d_weights

class DPN(nn.Module):
    def __init__ (self, input_features, total_nodes, output_nodes, use_min_weights = True, device=None):
        super(DPN, self).__init__()
        self.weights = nn.ParameterList()
        self.output_nodes = output_nodes
        self.input_features = input_features
        self.total_nodes = total_nodes
        self.use_min_weights = use_min_weights
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compile(self):
        if len(self.weights) == 0:
            weight_list, biases = DPN.get_weights(self.total_nodes, self.input_features, self.output_nodes, self.use_min_weights, self.device)
            self.weights.extend(weight_list)
            self.biases = biases

    @staticmethod
    def get_weights(n, X, output_size, min_blocks, device):
        #n: number of neurons
        #X: input size
        #min_blocks: Bool to determine whether to calculate minimum number of possible blocks or maximum

        #CALCULATE MINIMUM BLOCKS
        if min_blocks:
            weights = []
            if n == output_size: #if all the neurons are output neurons, create one block
                std_dev = torch.sqrt(torch.tensor(2 / X, dtype=torch.float)).to(device)
                weights.append(nn.Parameter(torch.randn(n, X).to(device) * std_dev))

            else: #if there are more neurons than output size, create two blocks

                std_dev = torch.tensor([2 / X, 2 / (X + n - output_size)], dtype=torch.float32)
                std_dev = torch.sqrt(std_dev).to(device)

                #First Block
                w1 = torch.randn(n - output_size, X).to(device)
                w1 *= std_dev[0]
                weights.append(nn.Parameter(w1))

                #Second Block
                weights.append(nn.Parameter(torch.randn(output_size, X + n - output_size).to(device) * std_dev[1]))

            biases = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0)).to(device)
            return weights, biases

        #CALCULATE MAXIMUM BLOCKS
        std_dev = torch.arange(X, X + n).view(-1, 1)
        std_dev = torch.sqrt(2 / std_dev).to(device)

        weights = []
        biases = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0)).to(device)

        #initialize weights for the rest of the blocks
        for i in range(n):
            weights.append(nn.Parameter(torch.randn(1, X + i).to(device) * std_dev[i]))
            

        return weights, biases
    
    def forward(self, x):

        return DPNAutoGrad.apply(x, self.output_nodes, self.total_nodes, self.biases, *list(self.weights))