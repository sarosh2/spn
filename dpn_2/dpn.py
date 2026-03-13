import torch
import torch.nn as nn
import torch.autograd as autograd

class DPNAutoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, input, output_size, biases, *weights):

        nodes = weights[0].shape[0]

        #calculate output of the first block separately as the input is different
        output = input.matmul(weights[0].T) + biases

        #calculate output for rest of the blocks
        for i in range(1, len(weights)):

            #get relavent output to use as input for next block
            input_start_idx = nodes - weights[i - 1].shape[0]
            input_end_idx = nodes - weights[i].shape[0]

            #apply relu to non output neurons
            if input_start_idx < nodes - output_size:
                output[:, input_start_idx : input_end_idx].relu_()

            #add to the partial outputs from the previous blocks
            output[:, input_end_idx:] += output[:, input_start_idx : input_end_idx].matmul(weights[i].T)

        ctx.output_size = output_size
        ctx.save_for_backward(input, output, *weights)
        return output[:, -output_size:]

    @staticmethod
    def backward(ctx, grad_output):

        #retrieve saved tensors
        saved = ctx.saved_tensors
        input = saved[0]
        output = saved[1]
        weights = saved[2:]

        #variables used for gradient calculation
        nodes = weights[0].shape[0]
        output_size = ctx.output_size
        batch_size = input.shape[0]

        #temp d_output tensor saves output gradients for each neuron as they're calculated
        d_output = input.new_zeros(batch_size, nodes)
        d_output[:, -output_size:] = grad_output

        d_weights = []

        #go through weight blocks except the first one in reverse
        for i in reversed(range(1, len(weights))):
            
            #get relavent indices for the input of the current block
            input_start_idx = nodes - weights[i - 1].shape[0]
            input_end_idx = nodes - weights[i].shape[0]

            d_o = d_output[:, input_end_idx:] #d_output for the current block

            d_weights.insert(0, d_o.T.matmul(output[:, input_start_idx : input_end_idx]) / batch_size) #add to the beginning of d_weights
            d_output[:, input_start_idx : input_end_idx] += d_o.matmul(weights[i])#update partial output gradients for prevoius block

            #if the previous input belongs to a non-output neuron block, apply relu gradient to it's d_output
            if input_start_idx < nodes - output_size:
                d_output[:, input_start_idx : input_end_idx] *= (output[:, input_start_idx : input_end_idx] > 0).float() #apply relu gradient

        #calculate for the first block separately as the input is different
        d_weights.insert(0, d_output.T.matmul(input) / batch_size)

        #finally calculate d_biases and d_input
        d_biases = d_output.sum(0) / batch_size
        d_input = d_output.matmul(weights[0])
              
        return d_input, None, d_biases, *d_weights

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
                w1 = torch.randn(n, X).to(device)
                w1[:-output_size , :] *= std_dev[0]
                w1[-output_size: , :] *= std_dev[1]
                weights.append(nn.Parameter(w1))

                #Second Block
                weights.append(nn.Parameter(torch.randn(output_size, n - output_size).to(device) * std_dev[1]))

            biases = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0)).to(device)
            return weights, biases

        #CALCULATE MAXIMUM BLOCKS

        #the first block has X features, the following blocks will have only 1


        std_dev = torch.arange(X, X + n).view(-1, 1)
        std_dev = torch.sqrt(2 / std_dev).to(device)

        biases = nn.Parameter(torch.empty(n).uniform_(0.0, 1.0)).to(device)

        weights = [nn.Parameter(torch.randn(n, X).to(device) * std_dev)]

        #initialize weights for the rest of the blocks
        for i in range(n - 1, 0, -1):
            weights.append(nn.Parameter(torch.randn(i, 1).to(device) * std_dev[-i:]))

        return weights, biases
    
    def forward(self, x):

        return DPNAutoGrad.apply(x, self.output_nodes, self.biases, *list(self.weights))