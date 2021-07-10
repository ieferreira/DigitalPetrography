batch_size = 20


import os

# list of image folders / classes s/"
ls = os.listdir()


ls[1]


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# reescalar imagenes

train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1/255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.8,
        horizontal_flip=True,
        fill_mode='nearest')

# partir en grupo de entrenamiento y de clasificacion
train_generator = train_datagen.flow_from_directory(
        '',  # directorio, aca
        target_size=(266, 400),  #reescalar
        batch_size=batch_size,
        # especificar clases obtenidas
        classes = ls,
        # usando las categorias
        class_mode='categorical',
        subset='training')

val_generator = train_datagen.flow_from_directory(

        '',  # Directorio, acá
        target_size=(266, 400),  # reescalar
        batch_size=batch_size,
        # especificar clases obtenidas
        classes = ls,
        # usando las categorias
        class_mode='categorical',
        subset='validation')
