class Earlystopping:
    def __init__(self, attemps):
        self.attemps = attemps
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        elif self.counter <= self.attemps:
            self.counter +=1
            return False
        elif self.counter > self.attemps:
            return True
        