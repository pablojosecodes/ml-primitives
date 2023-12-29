import numpy as np
import random


class Network:
    
    def __init__(self, sizes):
        self.biases = [np.random.randn(l,1) for l in sizes[1:]]
        self.weights = [np.random.randn(l,r) for r, l in zip(sizes[:-1], sizes[1:])]
        self.n = len(sizes)
    
    
    # Just runs through the weights and returns final layer's activations (result)
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = np.dot(w,a) + b
        return a
    
    def save(self, filename):
        data = {"weights": self.weights, "biases": self.biases}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        network = cls([1])  # Create a temporary network to replace weights and biases
        network.weights = data["weights"]
        network.biases = data["biases"]
        return network

    
    def mini_batch_update(self, batch, eta):
        
        # What we'll update our weights + biases with after computing
        # the gradient of cost function
        w_update = [np.zeros(w.shape) for w in self.weights]
        b_update = [np.zeros(b.shape) for b in self.biases]
        
        for x, y in batch:
            w,b = self.backprop(x,y)
            
            # small updates
            w_update = [w+nw for w,nw in zip(w_update, w)]
            b_update = [b+nb for b,nb in zip(b_update, b)]
        
        # Now update the actual parameters!
        
        self.weights = [w - (eta/len(batch)) * update for w, update in zip(self.weights, w_update) ]
        self.biases = [b - (eta/len(batch)) * update for b, update in zip(self.biases, b_update) ]
        
    # now for the most exciting part!!!
    def backprop(self, x, y):
        
        # Feedforward: we need all the activations and weighted inputs zs
        activation = x
        activations = [activation]
        zs = []
        
        w_nabla = [np.zeros(w.shape) for w in self.weights]
        b_nabla = [np.zeros(b.shape) for b in self.biases]
        
        
        for w,b in zip(self.weights, self.biases):
            
            z = np.dot(w,activation) + b
            
            
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
            
        # Final Layer
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        
        # Now we can update the nabla_b and nabla_b for final layer
        b_nabla[-1] = delta
        w_nabla[-1] = np.dot(delta, activations[-2].transpose())
    
        # Backprop
        for l in range(2,self.n):
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sigmoid_prime(zs[-l])
            b_nabla[-l] = delta
            w_nabla[-l] = np.dot(delta, activations[-l-1].transpose())
        return (w_nabla, b_nabla)
        
        
    # we're using a very simple cost function
    def cost_derivative(self, final_activation, actual_output):
        return (final_activation - actual_output)
        
        
    # Training!
    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[x:x+batch_size] for x in range(0,len(training_data), batch_size)]
            
            for batch in batches:
                self.mini_batch_update(batch, eta)
                
            if (test_data):
                print(f"Epoch {epoch}: {self.evaluate(test_data) / len(test_data)}")
                

    
    
    
    def evaluate(self, test_data):
        
        # x: input layer
        # y: right answer
        results = [(int(np.argmax(self.feedforward(x))), int(np.argmax(y))) for x, y in test_data]
        
#         print(type(results[0][0]))
#         print(type(results[0][1]))
        correct = 0
        for _ in range(len(results)):
            if (results[_][0]==results[_][1]):
                correct += 1
        return correct
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))
