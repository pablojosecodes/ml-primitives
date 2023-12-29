from neural_network import Network 
from mnist_loader import load


def main():
    (training_data, test_data) = load()
    nn = Network([784, 30, 10])
    nn.SGD(training_data, 30, batch_size=1000, eta=3, test_data=test_data)
    nn.evaluate(test_data) / len(test_data) 

if __name__ == "__main__":
    main()
