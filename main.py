import tensorflow as tf
from models import *
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.utils import get_custom_objects
from plotting_utils import *
from loss_metrics import *
from data import *
import config
from tqdm import tqdm

def train_step(model, training_dataset, optimizer, num_steps, loss_function, dice_metric, meanIOU_metric, current_steps, total_steps):
    loop = tqdm(training_dataset, leave=True)
    dice_list = []
    meanIOU_list = []
    loss_list = []
    for i, (img, true) in enumerate(loop):
        if i == num_steps:
            break

        lr = config.LEARNING_RATE_END + 0.5 * (config.LR - config.LEARNING_RATE_END) * ((1 + np.cos((current_steps) / (total_steps) * np.pi)))
        optimizer.learning_rate.assign(lr)

        with tf.GradientTape() as t:
            pred = model(img)
            loss = loss_function(true, pred)
        grads = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        dice_list.append(dice_metric(true, pred).numpy())
        meanIOU_list.append(meanIOU_metric(true, pred).numpy())
        loss_list.append(loss.numpy())

        current_steps += 1

        mean_loss = sum(loss_list) / len(loss_list)
        mean_dice = sum(dice_list) / len(dice_list)
        mean_IOU = sum(meanIOU_list) / len(meanIOU_list)

        loop.set_postfix(loss=mean_loss, dice=mean_dice, meanIOU=mean_IOU, lr=lr)

    return current_steps

def val_step(model, val_dataset, num_steps, loss_function, dice_metric, meanIOU_metric):
    loop = tqdm(val_dataset, leave=True)
    dice_list = []
    meanIOU_list = []
    loss_list = []
    for i, (img, true) in enumerate(loop):
        if i == num_steps:
            break

        pred = model(img)
        loss = loss_function(true, pred)
        dice_list.append(dice_metric(true, pred).numpy())
        meanIOU_list.append(meanIOU_metric(true, pred).numpy())
        loss_list.append(loss.numpy())

        mean_loss = sum(loss_list) / len(loss_list)
        mean_dice = sum(dice_list) / len(dice_list)
        mean_IOU = sum(meanIOU_list) / len(meanIOU_list)

        loop.set_postfix(loss=mean_loss, dice=mean_dice, meanIOU=mean_IOU)

def main():

    print('Begin')

    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Generating the training set
    train_inp, train_ground = load_list_from_df(config.TRAINING_PATH)
    training_list_ds = tf.data.Dataset.list_files(train_inp)
    training_images_ds = training_list_ds.map(training_image_label_pairs_processing, num_parallel_calls=config.AUTOTUNE)

    # Generates the validation set
    val_inp, val_ground = load_list_from_df(config.VAL_PATH)
    val_list_ds = tf.data.Dataset.list_files(val_inp)
    val_images_ds = val_list_ds.map(val_image_label_pairs_processing, num_parallel_calls=config.AUTOTUNE)

    training_prepared_dataset = preparation_for_training(training_images_ds, config.BATCH_SIZE)
    val_prepared_dataset = preparation_for_training(val_images_ds, config.BATCH_SIZE)

    # load model
    if config.SEGMENTOR == 'DEEPLABV3':
        model = build_deeplab3plus_model(config.HEIGHT, config.WIDTH, config.CHANNELS, num_classes=19, l2_scale=1e-3, network=config.NETWORK_BACKBONE)
    elif config.SEGMENTOR == 'UNET':
        model = build_unet_model(config.HEIGHT, config.WIDTH, config.CHANNELS, num_classes=19, l2_scale=1e-3, network=config.NETWORK_BACKBONE)
    else:
        model = build_swiftnet_model(config.HEIGHT, config.WIDTH, config.CHANNELS, num_classes=19, l2_scale=1e-3,
                                         network=config.NETWORK_BACKBONE)

    model.summary()

    total_steps = config.NUM_BATCH * config.EPOCHS
    current_step = 0

    loss_function = weighted_categorical_crossentropy(config.WEIGHTS)
    dice_metric = dice_coeff()
    meanIOU_metric = meanIOU()
    optimizer = tf.keras.optimizers.Adam(lr=config.LR)

    # training loop
    for i in range(config.EPOCHS):
        current_step = train_step(model, training_prepared_dataset, optimizer, config.NUM_BATCH, loss_function, dice_metric, meanIOU_metric, current_step, total_steps)
        val_step(model, val_prepared_dataset, config.NUM_BATCH // 100, loss_function, dice_metric, meanIOU_metric)

    model.save(config.SAVE_PATH + config.SAVE_NAME + '.h5')
    print("Saved model to disk")

if __name__ == '__main__':
    main()




