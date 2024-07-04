from keras import layers, Sequential


class ClassifierModel:
    
    def network(self):
        model = Sequential()

        model.add(layers.Conv2D(32,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(256, 256, 3)
                                ))
        model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Conv2D(64,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(256, 256, 3)
                                ))
        model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Conv2D(128,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(256, 256, 3)
                                ))
        model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Conv2D(256,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(256, 256, 3)
                                ))
        model.add(layers.MaxPool2D(2, 2))
        
        model.add(layers.Conv2D(512,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu',
                                input_shape=(256, 256, 3)
                                ))
        model.add(layers.MaxPool2D(2, 2))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

