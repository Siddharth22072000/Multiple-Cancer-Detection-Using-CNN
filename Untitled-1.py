# %%
from google.colab import files
files.upload()

# %%
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d obulisainaren/multi-cancer

# %%
!unzip '/content/multi-cancer.zip'

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

# %%
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers

# %%
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix

# %%
class ImageDataProcessor:
    def __init__(self, path):
        self.base_path = path
        self.train_datagen = ImageDataGenerator(validation_split=0.3)
        self.no_of_classes = 0
        self.class_names = []

    def initiate_generator(self):
        self._generate_dataset()
        self._generate_data_generators()
        self._plot_sample_images()
        self._print_image_shape()
        return self.class_names, self.no_of_classes, self.train_generator, self.validation_generator

    def _generate_dataset(self):
        print("\nTotal : ", end=" ")
        self.train_dataset = tf.keras.preprocessing.image_dataset_from_directory(batch_size=32, directory=self.base_path)
        self.class_names = self.train_dataset.class_names
        self.no_of_classes = len(self.class_names)

    def _generate_data_generators(self):
        self.train_generator = self._create_generator(subset='training')
        self.validation_generator = self._create_generator(subset='validation', shuffle=False)
        print("\nNo of Classes : ", self.no_of_classes)
        print("Classes : ", self.class_names)

    def _create_generator(self, subset, shuffle=True):
        print(f"\nFor {subset.capitalize()} : ", end=" ")
        return self.train_datagen.flow_from_directory(
            self.base_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset=subset,
            shuffle=shuffle
        )

    def _plot_sample_images(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_dataset.take(1):
            for i in range(self.no_of_classes):
                ax = plt.subplot(4, 4, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")

    def _print_image_shape(self):
        for image_batch, _ in self.train_dataset.take(1):
            print("Image Shape : ", image_batch.shape)
            break

# %%
class DataNormalizer:
    def __init__(self, train_generator, val_generator):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.normalized_ds = None
        self.AUTOTUNE = tf.data.AUTOTUNE

    def initiate_normalize(self):
        self._prepare_datasets()
        self._normalize_datasets()
        self._display_sample()

    def _prepare_datasets(self):
        self.train_ds = self.train_generator.cache().shuffle(1000).prefetch(buffer_size=self.AUTOTUNE)
        self.val_ds = self.val_generator.cache().prefetch(buffer_size=self.AUTOTUNE)

    def _normalize_datasets(self):
        normalization_layer = layers.Rescaling(1./255)
        self.normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))

    def _display_sample(self):
        image_batch, labels_batch = next(iter(self.normalized_ds))
        first_image = image_batch[0]
        print(f"Min pixel value: {np.min(first_image)}, Max pixel value: {np.max(first_image)}")


# %%
class ImageClassifier:
    def __init__(self, no_of_classes, image_size, class_name, train_generator, validation_generator):
        self.no_of_classes = no_of_classes
        self.image_size = image_size
        self.class_name = class_name
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.model = None
        self.annealer = None
        self.checkpoint = None

    def initiate_model(self):
        model_input = tf.keras.applications.VGG16(
            input_shape=self.image_size + [3],
            include_top=False,
            weights="imagenet"
        )

        for layer in model_input.layers:
            layer.trainable = False

        x = Flatten()(model_input.output)
        prediction = Dense(self.no_of_classes, activation='softmax')(x)

        self.model = Model(inputs=model_input.input, outputs=prediction)
        return self.model

    def model_summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model has not been initialized yet.")


    def initiate_params(self, lr):
        if self.model is None:
            print("Model has not been initialized yet.")
            return

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        self.annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
        self.checkpoint = ModelCheckpoint(self.class_name + 'VGG16.h5', verbose=1, save_best_only=True)

        return self.model, self.annealer, self.checkpoint

    def model_fit(self, epochs=20, batch_size=256):
        if self.model is None:
            print("Model has not been initialized yet.")
            return

        history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[self.annealer, self.checkpoint],
            steps_per_epoch=len(self.train_generator),
            validation_steps=len(self.validation_generator)
        )
        return history

    def eval_model(self):
        if self.model is None:
            print("Model has not been initialized yet.")
            return

        evl = self.model.evaluate(self.validation_generator)
        acc = evl[1] * 100
        msg = f'Accuracy on the Test Set = {acc:5.2f} %'
        print(msg)

    def save_model(self):
        if self.model is None:
            print("Model has not been initialized yet.")
            return

        file_path = self.class_name + " - VGG16.h5"
        self.model.save(file_path)
        print(f"Model saved to {file_path}!")



# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

class PlotMetrics:

    def plot_output(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs)

        sns.set(style='whitegrid')
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(epochs_range, acc, label='Training Accuracy')
        sns.lineplot(epochs_range, val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        sns.lineplot(epochs_range, loss, label='Training Loss')
        sns.lineplot(epochs_range, val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.tight_layout()
        plt.show()
        plt.savefig(self.class_name + '_performance_graph.png')

    def plot_confusion_matrix(self, cm, target_names, title='Confusion matrix', normalize=True):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', cbar=False,
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel(f'Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
        plt.tight_layout()
        plt.show()
        plt.savefig(title + '.png')

    def call_plot(self):
        y_true = self.validation_generator.classes
        y_pred = self.model.predict(self.validation_generator)
        y_pred = np.argmax(y_pred, axis=1)
        conf_mat = confusion_matrix(y_true, y_pred)

        self.plot_confusion_matrix(cm=conf_mat,
                                   normalize=False,
                                   target_names=self.class_names,
                                   title=self.class_name + " Confusion Matrix")


# %%
data_dir = '/content/Multi Cancer'
cancer_classes = os.listdir(data_dir)
print(cancer_classes)

# %% [markdown]
# #Cervical Caner

# %%
target_class = 'Cervical Cancer'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
cervical_model = classifierObj.initiate_model()

# %%
cervical_model, cervical_annealer, cervical_model_checkpoints = classifierObj.initiate_params(lr=0.001)
cervical_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

class PlotMetrics:
    def plot_output(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(1, epochs + 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
        sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
        sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

# Assuming `cervical_model_history` is a history object returned by the `.fit()` method of a Keras model,
# and you've trained for 10 epochs:
plot_metrics_obj = PlotMetrics()
plot_metrics_obj.plot_output(cervical_model_history, 10)


# %% [markdown]
# #Brain Cancer

# %%
target_class = 'Brain Cancer'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
brain_model = classifierObj.initiate_model()

# %%
brain_model, brain_annealer, brain_model_checkpoints = classifierObj.initiate_params(lr=0.001)
brain_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
def plot_training_history(history, epochs):
    # Extracting accuracy and loss values for plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Setting the range of epochs
    epochs_range = range(1, epochs + 1)

    # Plotting Training and Validation Accuracy
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Now, call this function with your history object and the number of epochs
plot_training_history(brain_model_history, 10)


# %% [markdown]
# #Kidney Cancer

# %%
target_class = 'Kidney Cancer'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
kidney_model = classifierObj.initiate_model()

# %%
kidney_model, kidney_annealer, kidney_model_checkpoints = classifierObj.initiate_params(lr=0.001)
kidney_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
def plot_training_history(history, epochs):
    # Extract accuracy and loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Prepare the range of epochs
    epochs_range = range(1, epochs + 1)

    # Create figure for plotting
    plt.figure(figsize=(14, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Call the function with your model's history and the number of epochs
plot_training_history(kidney_model_history, 10)


# %% [markdown]
# #Breast Cancer

# %%
target_class = 'Breast Cancer'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
breast_model = classifierObj.initiate_model()

# %%
breast_model, breast_annealer, breast_model_checkpoints = classifierObj.initiate_params(lr=0.001)
breast_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history, epochs):
    # Extracting accuracy and loss values for plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Setting the range of epochs
    epochs_range = range(1, epochs + 1)

    # Creating a figure for two subplots (2 rows, 1 column)
    plt.figure(figsize=(14, 6))

    # Plotting Training and Validation Accuracy
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    # Displaying the plot
    plt.tight_layout()
    plt.show()

# Now, call this function with your history object and the number of epochs you trained for
plot_training_history(breast_model_history, 10)


# %% [markdown]
# #Lung and Colon Cancer

# %%
target_class = 'Lung and Colon Cancer'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
lung_model = classifierObj.initiate_model()

# %%
lung_model, lung_annealer, lung_model_checkpoints = classifierObj.initiate_params(lr=0.001)
lung_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history, epochs):
    # Extracting accuracy and loss values for plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Setting the range of epochs
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 6))

    # Plotting Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Call the function with the history object and the number of epochs you've trained for
plot_training_history(lung_model_history, 10)


# %% [markdown]
# #Lymphoma

# %%
target_class = 'Lymphoma'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
lymph_model = classifierObj.initiate_model()

# %%
lymph_model, lymph_annealer, lymph_model_checkpoints = classifierObj.initiate_params(lr=0.001)
lymph_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history, epochs):
    # Extracting accuracy and loss values for plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Setting the range of epochs
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 6))

    # Plotting Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Now, call this function with your history object and the number of epochs you trained for
plot_training_history(lymph_model_history, 10)


# %% [markdown]
# #Oral Cancer

# %%
target_class = 'Oral Cancer'
target_data_path = f'/content/Multi Cancer/{target_class}'

dataProcessor = ImageDataProcessor(target_data_path)
classes, class_count, train_gen, valid_gen = dataProcessor.initiate_generator()

classifierObj = ImageClassifier(no_of_classes=class_count, class_name=target_class, image_size=[224, 224], train_generator=train_gen, validation_generator=valid_gen)
oral_model = classifierObj.initiate_model()

# %%
oral_model, oral_annealer, oral_model_checkpoints = classifierObj.initiate_params(lr=0.001)
oral_model_history = classifierObj.model_fit(epochs=10, batch_size=256)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history, epochs):
    # Extract accuracy and loss values from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Define the range of epochs
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 6))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x=epochs_range, y=acc, label='Training Accuracy')
    sns.lineplot(x=epochs_range, y=val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    sns.lineplot(x=epochs_range, y=loss, label='Training Loss')
    sns.lineplot(x=epochs_range, y=val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Use this function with your oral cancer model's training history and the number of epochs
plot_training_history(oral_model_history, 10)


# %%
import pandas as pd

# Assuming all history objects follow the naming convention "{cancer_type}_model_history"
# and have been defined in your workspace.

model_histories = {
    "Cervical": cervical_model_history,
    "Brain": brain_model_history,
    "Kidney": kidney_model_history,
    "Breast": breast_model_history,
    "Lung_and_Colon": lung_model_history,
    "Lymphoma": lymph_model_history,
    "Oral": oral_model_history
}

# Initialize an empty list to store epoch data
epoch_data = []

# Loop through each model's history to extract epoch data
for cancer_type, history in model_histories.items():
    for epoch in range(10):  # Assuming all models were trained for 10 epochs
        epoch_data.append({
            "Cancer Type": cancer_type,
            "Epoch": epoch + 1,
            "Training Accuracy": history.history['accuracy'][epoch],
            "Validation Accuracy": history.history['val_accuracy'][epoch],
            "Training Loss": history.history['loss'][epoch],
            "Validation Loss": history.history['val_loss'][epoch]
        })

# Create a DataFrame from the epoch data
epoch_df = pd.DataFrame(epoch_data)

# Display the DataFrame
print(epoch_df)


# %%
import pandas as pd

# Assuming all the history objects are named according to the cancer type followed by '_model_history'
# and all models have completed at least 10 epochs of training.

model_histories = {
    "Cervical": cervical_model_history,
    "Brain": brain_model_history,
    "Kidney": kidney_model_history,
    "Breast": breast_model_history,
    "Lung_and_Colon": lung_model_history,
    "Lymphoma": lymph_model_history,
    "Oral": oral_model_history
}

# Initialize an empty list to store the 10th epoch performance data for each model
epoch_10_data = []

# Extract the 10th epoch data from each model's history
for cancer_type, history in model_histories.items():
    epoch_10_data.append({
        "Cancer Type": cancer_type,
        "Training Accuracy": history.history['accuracy'][-1],  # Last epoch accuracy
        "Validation Accuracy": history.history['val_accuracy'][-1],  # Last epoch validation accuracy
        "Training Loss": history.history['loss'][-1],  # Last epoch loss
        "Validation Loss": history.history['val_loss'][-1]  # Last epoch validation loss
    })

# Convert the list of data into a Pandas DataFrame
epoch_10_df = pd.DataFrame(epoch_10_data)

# Sorting the DataFrame based on a specific column, e.g., 'Validation Accuracy', in descending order
epoch_10_df.sort_values(by='Validation Accuracy', ascending=False, inplace=True)

# Resetting the index of the DataFrame for better readability
epoch_10_df.reset_index(drop=True, inplace=True)

# Display the DataFrame
print(epoch_10_df)


# %%
import pandas as pd
from tabulate import tabulate

# Assuming the DataFrame 'epoch_10_df' is already created as before

# Define a function for formatting the values to a specified number of decimal places
def format_dataframe(df, decimal_places=4):
    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column] = df[column].map(lambda x: f'{x:.{decimal_places}f}')
    return df

# Format the DataFrame
epoch_10_df_formatted = format_dataframe(epoch_10_df.copy(), 4)

# Display the DataFrame using tabulate for a well-defined text table
print(tabulate(epoch_10_df_formatted, headers='keys', tablefmt='pretty', showindex=False))


# %%
import pandas as pd

# Assuming all history objects follow the naming convention "{cancer_type}_model_history"
# and have been defined in your workspace.

model_histories = {
    "Cervical": cervical_model_history,
    "Brain": brain_model_history,
    "Kidney": kidney_model_history,
    "Breast": breast_model_history,
    "Lung_and_Colon": lung_model_history,
    "Lymphoma": lymph_model_history,
    "Oral": oral_model_history
}

# Initialize an empty list to store data for the 10th epoch
epoch_10_data = []

# Extract data for the 10th epoch
for cancer_type, history in model_histories.items():
    epoch_10_data.append({
        "Cancer Type": cancer_type,
        "Training Accuracy": history.history['accuracy'][9],  # 10th epoch, zero-indexed
        "Validation Accuracy": history.history['val_accuracy'][9],
        "Training Loss": history.history['loss'][9],
        "Validation Loss": history.history['val_loss'][9]
    })

# Create a DataFrame from the 10th epoch data
epoch_10_df = pd.DataFrame(epoch_10_data)

# Reorder DataFrame columns if desired
epoch_10_df = epoch_10_df[['Cancer Type', 'Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss']]




# %%
print(epoch_10_df)


# %%
# prompt: print(epoch_10_df)
# make the printed table better

# Assuming the DataFrame 'epoch_10_df' is already created as before

# Define a function for formatting the values to a specified number of decimal places
def format_dataframe(df, decimal_places=4):
    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column] = df[column].map(lambda x: f'{x:.{decimal_places}f}')
    return df

# Format the DataFrame
epoch_10_df_formatted = format_dataframe(epoch_10_df.copy(), 4)

# Display the DataFrame using tabulate for a well-defined text table
print(tabulate(epoch_10_df_formatted, headers='keys', tablefmt='pretty', showindex=False))


# %%
# prompt: suggest graph for above table

import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data for the bar plot
df_bar = epoch_10_df.melt(id_vars='Cancer Type', var_name='Metric', value_name='Value')

# Create the bar plot with Seaborn
sns.barplot(x='Cancer Type', y='Value', hue='Metric', data=df_bar)
plt.title('Model Performance on Different Cancer Types (10th Epoch)')
plt.xlabel('Cancer Type')
plt.ylabel('Metric Value')
plt.xticks(rotation=45)
plt.show()


# %%
# Export the DataFrame to an Excel file
excel_filename = "/content/cancer_model_epoch_10_summary.xlsx"
epoch_10_df.to_excel(excel_filename, index=False)

print(f"Summary table for the 10th epoch has been saved to {excel_filename}")

# %%
# prompt: save excel file in drive

from google.colab import drive

# Mount your Google Drive
drive.mount('/content/drive')

# Specify the path to your Excel file within your Google Drive
excel_file_path = '/content/drive/My Drive/cancer_model_epoch_10_summary.xlsx'

# Save the DataFrame to the specified path
epoch_10_df.to_excel(excel_file_path, index=False)

print(f"Summary table for the 10th epoch has been saved to {excel_file_path}")



