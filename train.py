from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = 'dataset'
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
BATCH_SIZE = 8
FREEZE_LAYERS = 2


def train_model():
    train_data_generator = ImageDataGenerator(rotation_range=40,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              channel_shift_range=10,
                                              horizontal_flip=True,
                                              fill_mode='nearest')

    train_batches = train_data_generator.flow_from_directory(DATASET_PATH + '/train',
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      class_mode='categorical',
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE)

    valid_datagen = ImageDataGenerator()
    valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                      target_size=IMAGE_SIZE,
                                                      interpolation='bicubic',
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                      batch_size=BATCH_SIZE)

    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                   input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    final_model = Model(inputs=net.input, outputs=output_layer)
    for layer in final_model.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in final_model.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    final_model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    history = final_model.fit(train_batches, steps_per_epoch=120, epochs=30, verbose=2, validation_data=valid_batches,
                            validation_steps=40)

    final_model.save('model_1217.h5')

    return history


#h = train_model()
