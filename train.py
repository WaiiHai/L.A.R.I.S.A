import tensorflow as tf

import model as nn

def main():
    BATCH_SIZE = 8

    from dataloader import DataLoader, RandomCrop, RandomFlip, RandomRotation
    (train_data, train_len) = DataLoader('data/train')(batch_size=BATCH_SIZE, transforms=[RandomCrop(),
                                                                                          RandomFlip(), 
                                                                                          RandomRotation()])
    (valid_data, valid_len) = DataLoader('data/test')(batch_size=BATCH_SIZE)

    (image, mask) = next(iter(train_data))

    # from utils import display
    # display([image, mask], 4)

    from model import build_unet, build_deeplabv3plus, build_unet3plus, build_efficientnet_unet
    model = build_unet((256, 256, 3))

    from keras.optimizers import Adam
    model_optimizer = Adam(1e-5)

    from metrics import dice_loss, dice_coef
    model.compile(optimizer=model_optimizer, loss=dice_loss, metrics=[dice_coef])

    from logger import Logger
    logger = Logger('logs')
    path = logger(model.name)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=5,),
                 tf.keras.callbacks.ModelCheckpoint(filepath=f'{path}/model.weights.h5', save_weights_only=True,
                                                    monitor='val_loss', mode='min', save_best_only=True),]
    history = model.fit(train_data, validation_data=(valid_data), epochs=100, callbacks=callbacks)

    
    model.load_weights(f'{path}/model.weights.h5')
    (val_loss, val_accuracy) = model.evaluate(valid_data)

    from logger import Report
    report = Report(model.name, 'DiceLoss',
                    train_len, valid_len, BATCH_SIZE,
                    len(history.epoch), val_loss, val_accuracy)
    logger.save_results(report)

    logger.save_plot((history.history['loss'], history.history['val_loss']), 'loss_graphics')
    logger.save_plot((history.history['dice_coef'], history.history['val_dice_coef']), 'accuracy_graphics')

    (image, mask) = next(iter(valid_data))
    logger.save_predict([image, mask, model.predict(image)])

if __name__ == '__main__':
    main()
    tf.keras.backend.clear_session()