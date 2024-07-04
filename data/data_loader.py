import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


class DataLoader:

    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir

    def load_data(self):
        # normalizing image in the range of 0 and 1
        data_generator = ImageDataGenerator(rescale=1./255)
        
        #custom train generator for augmentation
        train_gen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 15,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            zoom_range = 0.1,
            horizontal_flip = True, 
            fill_mode = 'nearest'
        )

        # Loading data using custom generator
        train_generator = train_gen.flow_from_directory(
            directory=self.base_dir + 'train',
            target_size=(256, 256),
            batch_size=32,
            shuffle=True,
            class_mode='categorical'
        )

        test_generator = data_generator.flow_from_directory(
            directory=self.base_dir + 'test',
            target_size=(256, 256),
            batch_size=32,
            shuffle=True,
            class_mode='categorical'
        )

        val_generator = data_generator.flow_from_directory(
            directory=self.base_dir + 'val',
            target_size=(256, 256),
            batch_size=32,
            shuffle=True,
            class_mode='categorical'
        )

        return train_generator, test_generator, val_generator
