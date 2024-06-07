import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from preprocess import Dataset_Reader
from model import NoteClassificationModel
import argparse
import pickle
import numpy as np
import os

def visualize_data(datasets):
    # visualize the distribution of the classes
    plt.figure(figsize=(10, 6))
    sns.countplot(x=datasets.annotations)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

    # visualize some example images with their labels
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    for i in range(10):
        axes[i].imshow(datasets.images[i], cmap='gray')
        axes[i].set_title(f"Label: {datasets.annotations[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def train(model, datasets):
    # create the logs directory
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # initialize Tensorboard callback with log_dir
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./checkpoints/" +
            "weights.e{epoch:02d}-" +
            "acc{val_sparse_categorical_accuracy:.4f}.weights.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
            tf.keras.callbacks.CSVLogger('training_log.csv'),
            tensorboard_callback
    ]

    history = model.fit(
        x=np.array(datasets.images),
        y=datasets.annotations,
        validation_data=(np.array(datasets.test_images),
                         datasets.test_annotations),
        epochs=model.epochs,
        batch_size=model.batch_size,  # none for right now
        callbacks=callback_list,
        shuffle=True
    )

    # plot training history
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # predict and plot confusion matrix
    y_pred = model.predict(np.array(datasets.test_images))
    y_pred_classes = np.argmax(y_pred, axis=1)
    plot_confusion_matrix(datasets.test_annotations, y_pred_classes, datasets.class_names)

def test(model, datasets):
    model.evaluate(
        x=np.array(datasets.test_images),
        y=datasets.test_annotations,
        verbose=1,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="OMR TIME")
    parser.add_argument(
        '--old_data',
        action='store_true',
        help='''Loads old preprocess''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')

    return parser.parse_args()


def main():
    data_reader = Dataset_Reader("./dataset")
    data_reader.one_hot = False
    if not ARGS.old_data:
        print("Reading new data")
        data_reader.read_images()
        print("Saving new data")
    else:
        print("Loading old data")

    print("Shape of images")
    print(data_reader.images.shape)
    print("Entry of images")
    print(data_reader.images[0])
    print("Shape of annotations")
    print(data_reader.annotations.shape)
    print("Entry of annotations")
    print(data_reader.annotations[0])

    visualize_data(data_reader)

    model = NoteClassificationModel(data_reader.nr_classes)
    model(tf.keras.Input(
        shape=(data_reader.tile_size[0], data_reader.tile_size[1], 1)))
    model.summary()

    if ARGS.load_checkpoint is not None:
        print("Loading checkpoint")
        model.load_weights(ARGS.load_checkpoint)

    print("Compile model")
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        print("Start testing")
        test(model, data_reader)
    else:
        print("Start training")
        train(model, data_reader)

ARGS = parse_args()
main()
