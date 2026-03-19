    return model

def Discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=INPUT_IMG))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

generator = Generator()
generator.summary()
discriminator = Discriminator()
discriminator.summary()
from PIL import Image
import imageio
import cv2
from keras.preprocessing.image import img_to_array

# Библиотека линейной алгебры
import numpy as np

# Работа с файловой системой и системными функциями
import os

# Подготовка данных
from keras import preprocessing
import tensorflow as tf

# Построение сети
from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape

# Визуализация выполнения процессов, циклов
from tqdm import tqdm

# Генерация случайных чисел
import re

# Работа с графиками
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Отключаем лишние предупреждения
import warnings
warnings.filterwarnings('ignore')

#Импорт датасета
import gdown
gdown.download('https://drive.google.com/uc?export=download&id=1Epwm1nAZ62Hfn-kI9vnwyfJgKUJ2gAM6')

import zipfile
import os

# Создаем папку dataset, если её нет
os.makedirs("./dataset", exist_ok=True)

# Распаковываем архив
with zipfile.ZipFile("/content/face_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("./dataset")

data_dir = '/content/dataset/content/new/'
path_face = []
for path in os.listdir(data_dir):
    if '.png' in path:
        path_face.append(os.path.join(data_dir, path))
        

SIZE = 128                  # линейный размер изображений (одно измерение)
INPUT_IMG = (SIZE, SIZE, 3) # размер входного изображений
BATCH_SIZE = 32             # размер пакета для обучения
LATENT_DIM = 100            # размерность латентного пространства

images = [np.array((Image.open(path)).resize((SIZE, SIZE))) for path in path_face]

for i in range(len(images)):
    images[i] = images[i].astype(float)
    images[i] = (images[i] - 127.5) / 127.5

X_train = np.array(images)

del images, path_face

dataset=tf.data.Dataset.from_tensor_slices(X_train).batch(BATCH_SIZE)

def plot_images(sqr = 5):
    plt.figure(figsize = (10,10))
    plt.title("Реальные изображения", fontsize = 35)
    for i in range(sqr * sqr):
        plt.subplot(sqr,sqr,i+1)
        plt.imshow(X_train[i]*0.5 + 0.5 )
        plt.axis('off')

# to plot images
plot_images(6)


def Generator():
    model = Sequential()
    model.add(Dense(SIZE*SIZE*3, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(Reshape(INPUT_IMG))
    # downsampling (понижающая дискретизация)
    model.add(tf.keras.layers.Conv2D(SIZE,4, strides=1, padding='same',kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.Conv2D(SIZE,4, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(SIZE*2,4, strides=1, padding='same',kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.Conv2D(SIZE*2,4, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(SIZE*4, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.Conv2D(SIZE*4,4, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))

    model.add(tf.keras.layers.LeakyReLU())
    #upsampling (повышающая дискретизация)
    model.add(tf.keras.layers.Conv2DTranspose(SIZE*4, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(SIZE*4, 4, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(SIZE*2, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(SIZE*2, 4, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2DTranspose(SIZE, 4, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(SIZE, 4, strides=1,padding='same',kernel_initializer='he_normal',use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(3,4,strides = 1, padding = 'same',activation = 'tanh'))

    return model

generator = Generator()
generator.summary()

noise = np.random.normal(-1,1,(1,LATENT_DIM))
img = generator(noise)
plt.imshow(img[0,:,:,0])
plt.show()

optimizer_G = tf.keras.optimizers.RMSprop(learning_rate=.0001)
optimizer_D = tf.keras.optimizers.RMSprop(learning_rate=.0001)

# Этот метод возвращает вспомогательную функцию для вычисления перекрестных потерь энтропии
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(fake_output, real_output):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return fake_loss + real_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Мы используем декоратор `tf.function`, можно и без него
# Он указывает на то, что функция должна быть скомпилирована, что оптимизирует процесс обучения.
@tf.function
def train_steps(images):

    # генерируем шум на входе генератора
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

    # Инициализируем GradientTape (градиентную ленту) для дискриминатора и генератор
    #  по отдельности, чтобы записать операции,
    # выполняемые во время прямого прохода, включающего автодифференцирование.
    with tf.GradientTape() as gen_tape , tf.GradientTape() as disc_tape:
        # Запустим прямой проход шума через генератор.
        # Операции применяемые каждым слоем к своим
        # входным данным будут записаны на GradientTape, как на ленту магнитофона.
        # Отсюда и название градиентной ленты.
        generated_images = generator(noise)

        # Также обеспечиваем прямой проход через дискриминатор для
        # для реальных и фейковых изображений
        fake_output = discriminator(generated_images)
        real_output = discriminator(images)

        # расчитываем функции ошибок (значения потерь)
        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(fake_output, real_output)

    # Используем gradient tape для автоматического извлечения градиентов
    # обучаемых переменных относительно потерь.
    gradient_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)

    # Выполним один шаг градиентного спуска, обновив
    # значение переменных минимизирующих потери.
    optimizer_G.apply_gradients(zip(gradient_of_generator,generator.trainable_variables))
    optimizer_D.apply_gradients(zip(gradient_of_discriminator, discriminator.trainable_variables))

    # Вернем значения потерь для визуализации
    loss = {'gen_loss':gen_loss,
           'disc_loss': dis_loss}
    return loss

import time
def train(epochs, dataset):
    D_loss=[] # список для сбора значений потерь для дискриминатора
    G_loss=[] # список для сбора значений потерь для генератора
    for epoch in range(epochs):
        start = time.time()
        print("\nЭпоха : {}".format(epoch + 1))
        for images in dataset:
            loss = train_steps(images)
        print(" Время:{}".format(np.round(time.time() - start),2))
        print("Generator Loss: {} Discriminator Loss: {}".format(loss['gen_loss'],loss['disc_loss']))
        D_loss.append(loss['disc_loss'])
        G_loss.append(loss['gen_loss'])
    return (G_loss, D_loss)

G, D = train(10, dataset)

def plot_generated_images(square = 5):


  plt.figure(figsize = (10,10))
  for i in range(square * square):
    plt.subplot(square, square, i+1)
    noise = np.random.normal(0,1,(1,LATENT_DIM))
    img = generator(noise)
    plt.imshow(np.clip((img[0,...]+1)/2, 0, 1))

    plt.axis('off')
    plt.grid()

    plot_generated_images(5)

    def loss_plot(G, D):
        plt.figure(figsize=(10,10))
        plt.plot(G,color='red',label='Функция потерь на генераторе')
        plt.plot(D,color='blue',label='Функция потерь на дискриминаторе')
        plt.legend()
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.title('Функции потерь в конце обучения на эпохе')
        plt.show()
        
loss_plot(G, D)