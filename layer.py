class Layer:
    def __init__(self) -> None:
        self.input = None
        self.outout = None
    
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass