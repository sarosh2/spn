import cupy as cp
import math
import time
cp.random.seed(42)


class DPN:
    def __init__ (self, input_features, total_nodes, output_nodes):
        self.weights = None
        self.output_size = output_nodes
        self.input_size = input_features
        self.num_nodes = total_nodes
        self.input_features_to_remove = None

    def compile(self):
        if self.weights is None:
            self.weights = DPN.get_weights(self.num_nodes, self.input_size)
        
        self.blocks = self.get_blocks()
        self.m = cp.zeros_like(self.weights)
        self.v = cp.zeros_like(self.weights)  # Second moment vector

    def set_weights(self, weights):
        self.weights = weights

    def get_blocks(self):
        
        actual_thresholds = cp.argmax(self.weights[:, ::-1] != 0, axis=1)
        actual_thresholds = self.weights.shape[1] - actual_thresholds
        max_thresholds = cp.arange(self.input_size + 1, self.weights.shape[1] + 1)

        blocks = [0]
        current_threshold = 0

        for i in range(len(actual_thresholds)):
            if actual_thresholds[i] > max_thresholds[current_threshold]:
                blocks.append(i)
                current_threshold = i

        return blocks

    def get_weights(n, X):
        #X: input size
        #n: number of neurons

        weights = None
        param_count = 0

        #initialize weights for each neuron
        for i in range(n):
            input_size = X + i
            param_count += input_size + 1
            std_dev = math.sqrt(2 / input_size)
            w = cp.random.randn(1, input_size) * std_dev

            #pad with bias on left and zeros on right
            w = cp.pad(w, pad_width=((0, 0), (1, n - 1 - i)), mode='constant', constant_values=((float)(cp.random.uniform(0.0, 1.0)), 0))
            weights = w if weights is None else cp.vstack([weights, w])

        print("Parameters:", param_count)
        return weights
    
    def forward_prop(self, input):

        blocks = self.blocks
        input_size = self.input_size
        W = self.weights[:, :input_size + 1]
        output = cp.dot(W, input)
        n = self.num_nodes

        for i in range(len(blocks) - 1):
            
            #get relevant weights from the weights matrix
            column = input_size + 1 + blocks[i]
            weight_size = n - blocks[i + 1]
            W = self.weights[-weight_size:, column: input_size + 1 + blocks[i + 1]]
            
            #check if current output needs activation
            if blocks[i] < n - self.output_size:
                #Apply RELU to output
                output[blocks[i]:blocks[i + 1], :] = cp.maximum(0, output[blocks[i]:blocks[i + 1], :])

            #get relevant X output from output
            X = output[blocks[i]:blocks[i + 1], :]
            
            #get the product of W.X
            Z = cp.dot(W, X)

            #Add the product to the output of the next neurons
            output[blocks[i + 1]:, :] += Z
            
        return output
    
    def softmax(x):
        e_x = cp.exp(x - cp.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
        return e_x / cp.sum(e_x, axis=0, keepdims=True)

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
    
    def get_update(self, alpha, beta_1, beta_2, epsilon, t, dW, start, end):
        # Update biased first moment estimate
        self.m[start: end, :dW.shape[1]] = beta_1 * self.m[start: end, :dW.shape[1]] + (1 - beta_1) * dW
        # Update biased second raw moment estimate
        self.v[start: end, :dW.shape[1]] = beta_2 * self.v[start: end, :dW.shape[1]] + (1 - beta_2) * (dW ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m[start: end, :dW.shape[1]] / (1 - beta_1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[start: end, :dW.shape[1]] / (1 - beta_2 ** t)

        return alpha * m_hat / (cp.sqrt(v_hat) + epsilon)
    
    def get_batches(X, Y, batch_size, num_samples):
        """
        Generator that yields batches from input matrix X and labels Y.
        X: Input data of shape (n_features, n_samples)
        Y: Labels of shape (n_classes, n_samples)
        batch_size: Number of samples per batch
        """
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            yield X[:, start_idx:end_idx], Y[:, start_idx:end_idx]  # Yield batches of X and Y

    def back_prop(self, total_inputs, final_output, Y, alpha, beta_1, beta_2, epsilon, t):
        gradients = final_output - Y

        num_samples = gradients.shape[1]
        n = self.num_nodes
        blocks = self.blocks
        input_size = self.input_size

        #calculate for the final block separately
        d_output = gradients[-(n - blocks[-1]):, :]


        dW = cp.dot(d_output, total_inputs.T) / num_samples
        
        #calculate for subsequent arrays
        d_output = cp.dot(self.weights[blocks[-1]:, 1: input_size + 1 + blocks[-1]].T, d_output)

        self.weights[blocks[-1]:, :input_size + 1 + blocks[-1]] -= self.get_update(alpha, beta_1, beta_2, epsilon, t, dW, blocks[-1], n)

        for i in range(len(blocks) - 2, -1, -1):

            #for non output blocks
            if blocks[i] < n - self.output_size:
                d_o = d_output[input_size + blocks[i]: input_size + blocks[i + 1], :] * (total_inputs[input_size + 1 + blocks[i]: input_size + 1 + blocks[i + 1], :] > 0)

            #for output blocks
            else:
                d_o = gradients[-(n - blocks[i]): -(n - blocks[i + 1]), :] + d_output[input_size + blocks[i]: input_size + blocks[i + 1], :]
            
            dW = cp.dot(d_o, total_inputs[:input_size + 1 + blocks[i], :].T) / num_samples
            d_output[:input_size + blocks[i], :] += cp.dot(self.weights[blocks[i]: blocks[i + 1], 1: input_size + 1 + blocks[i]].T, d_o)
            self.weights[blocks[i]: blocks[i + 1], : 1 + input_size + blocks[i]] -= self.get_update(alpha, beta_1, beta_2, epsilon, t, dW, blocks[i], blocks[i + 1])

    def run_epoch(self, X, Y, alpha, beta_1, beta_2, epsilon, t):
        
        start = time.time()
        
        outputs = self.forward_prop(X)

        total_inputs = cp.vstack([X, outputs[:self.blocks[-1], :]])

        final_output = DPN.softmax(outputs[-self.output_size:, :])

        self.back_prop(total_inputs, final_output, Y, alpha, beta_1, beta_2, epsilon, t)

        end = time.time()

        #calculate loss
        loss = DPN.categorical_crossentropy(final_output, Y)

        #calculate accuracy
        accuracy = DPN.caclulate_accuracy(final_output, Y)

        return cp.hstack((end - start, cp.mean(loss), cp.mean(accuracy)))
    
    def fit(self, X_train, Y_train, X_val, Y_val, epochs, batch_size, alpha, beta_1, beta_2, epsilon):

        if self.input_features_to_remove is not None:
            X_train = X_train[~self.input_features_to_remove, :]
            X_val = X_val[~self.input_features_to_remove, :]
        
        train_metrics = []
        val_metrics = []

        t = 1 #timestep
        val_loss = 0
        val_accuracy = 0
        for i in range(epochs):
        
            metrics = []
            for batch_num, (batch_X, batch_Y) in enumerate(DPN.get_batches(X_train, Y_train, batch_size, X_train.shape[1])):
                metrics.append(self.run_epoch(batch_X, batch_Y, alpha, beta_1, beta_2, epsilon, t))
                t += 1

            metrics = cp.vstack(metrics)
            
            #validate output
            outputs = self.forward_prop(X_val)
            final_output = DPN.softmax(outputs[-10:, :])
            val_loss = cp.mean(DPN.categorical_crossentropy(final_output, Y_val))
            val_accuracy = cp.mean(DPN.caclulate_accuracy(final_output, Y_val))

            print(f"Epoch: {i + 1} Total_Time: {cp.sum(metrics[:, 0]):.4f} Average_Time_per_batch: {cp.mean(metrics[:, 0]):.4f} Train_Accuracy: {metrics[-1, 2]:.4f} Train_Loss: {metrics[-1, 1]:.4f} Val_Accuracy: {val_accuracy:.4f} Val_Loss: {val_loss:.4f}")
            train_metrics.append(metrics)
            val_metrics.append([val_loss, val_accuracy])

        return train_metrics, val_metrics

    def test(self, X_test, Y_test):

        if self.input_features_to_remove is not None:
            X_test = X_test[~self.input_features_to_remove, :]

        outputs = self.forward_prop(X_test)
        final_output = DPN.softmax(outputs[-self.output_size:, :])
        test_loss = cp.mean(DPN.categorical_crossentropy(final_output, Y_test))
        test_accuracy = cp.mean(DPN.caclulate_accuracy(final_output, Y_test))

        print("Test_Accuracy: ", test_accuracy, "Test_Loss: ", test_loss)
        return [test_loss, test_accuracy]


    def prune_by_percent_once(percent, mask, final_weight):
        # Put the weights that aren't masked out in sorted order.
        sorted_weights = cp.sort(cp.abs(final_weight[mask == 1]))

        # Determine the cutoff for weights to be pruned.
        cutoff_index = cp.round(percent * sorted_weights.size).astype(int)
        cutoff = sorted_weights[cutoff_index]

        # Prune all weights below the cutoff.
        return cp.where(cp.abs(final_weight) <= cutoff, cp.zeros(mask.shape), mask)
    
    def prune_by_percent(self, percent, masks):

        blocks = self.blocks.copy()
        blocks.append(self.num_nodes)
        #prune non-output weights
        for i in range(len(blocks) - 1):
            W = self.weights[blocks[i]: blocks[i + 1], 1: self.input_size + 1 + blocks[i]]
            mask = masks[blocks[i]: blocks[i + 1], : self.input_size + blocks[i]]
            if blocks[i] < self.num_nodes - self.output_size:
                p = percent
            else:
                p = percent / 2

            masks[blocks[i]: blocks[i + 1], : self.input_size + blocks[i]] = DPN.prune_by_percent_once(p, mask, W)

        return masks
        

    def apply_lth(self, percent, rounds, X_train, Y_train, X_val, Y_val, batch_size, alpha, beta_1, beta_2, epsilon):

        blocks = self.blocks.copy()
        blocks.append(self.num_nodes)

        original_weights = self.weights.copy()

        masks = []
        for i in range(len(blocks) - 1):
            #create mask
            mask = cp.ones((blocks[i + 1] - blocks[i], self.input_size + blocks[i]))
            masks.append(mask)

        max_pad = masks[-1].shape[1]
        for i in range(len(masks)):
            masks[i] = cp.pad(masks[i],  pad_width=((0, 0), (0, max_pad - masks[i].shape[1])), mode='constant', constant_values=(0))

        masks = cp.vstack(masks)

        p_per_round = 1 - (1 - percent) ** (1 / rounds)
        val_accuracy = self.fit(X_train, Y_train, X_val, Y_val, 1, batch_size, alpha, beta_1, beta_2, epsilon)[1]
        final_masks = masks.copy()
        
        for round in range(rounds):

            #update masks by pruning
            masks = self.prune_by_percent(p_per_round, masks)
            
            #reset weights and moment vectors
            pruned_weights = original_weights.copy()
            pruned_weights[:, 1:] *= masks
            self.set_weights(pruned_weights)
            self.compile()
            print(self.blocks)
            
            #train for one epoch
            new_val_accuracy = self.fit(X_train, Y_train, X_val, Y_val, 1, batch_size, alpha, beta_1, beta_2, epsilon)[1]

            if f"{new_val_accuracy:.4f}" >= f"{val_accuracy:.4f}":
                val_accuracy = new_val_accuracy
                final_masks = masks.copy()

        original_weights[:, 1:] *= final_masks

        '''empty_columns = cp.all(original_weights == 0, axis=0)
        self.input_features_to_remove = empty_columns[:self.input_size + 1]
        non_output_neuron_features_to_remove = empty_columns[self.input_size + 1: self.input_size + 1 + self.num_nodes - self.output_size]
        
        output_neuron_features_to_remove = empty_columns[self.input_size + 1 + self.num_nodes - self.output_size:]

        #not removing columns or rows pertaining to the output neurons as even though their columns are zero, the placement of those columns helps determines blocks and other calculations

        if output_neuron_features_to_remove.any():
            last_false_idx = cp.where(output_neuron_features_to_remove == False)[0][-1]
            output_neuron_features_to_remove[:last_false_idx] = False

        neurons_to_remove = cp.concatenate((non_output_neuron_features_to_remove, cp.full(self.output_size, False, dtype=bool)))
        original_weights = original_weights[~neurons_to_remove, :]
        
        empty_columns_to_remove = cp.concatenate((self.input_features_to_remove, non_output_neuron_features_to_remove, output_neuron_features_to_remove))
        
        original_weights = original_weights[:, ~empty_columns_to_remove]
        self.input_size -= (int)(cp.sum(self.input_features_to_remove))
        self.num_nodes -= (int)(cp.sum(non_output_neuron_features_to_remove))'''

        self.set_weights(original_weights)
        self.compile()
        print(self.blocks)