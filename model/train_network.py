

class TrainNetwork:
    def __init__(self, train, val) -> None:
        self.train = train 
        self.val = val
    
    def train_model(self, model):
        # model = self.model
        model.fit(
            x = self.train,
            batch_size = 32,
            epochs = 10,
            validation_data = self.val,
            verbose = 2
        )
        model.summary()
        return model