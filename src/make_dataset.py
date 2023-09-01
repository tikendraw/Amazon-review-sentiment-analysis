# train.py
import sys

import tensorflow as tf

# def create_datasets(x_train, y_train, text_vectorizer, batch_size):
#     print('Building slices...')
#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
#     print('Mapping...')
#     train_dataset = train_dataset.map(lambda x, y: (text_vectorizer(x), y), tf.data.AUTOTUNE)
#     print('Prefetching...')    
#     train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
#     return train_dataset

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                          locals().items())), key= lambda x: -x[1], reverse = False)[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))



def data_generator(x, y):
    num_samples = len(x)
    for i in range(num_samples):
        yield x[i], y[i]


def create_datasets(x, y, text_vectorizer, batch_size:int = 32, shuffle:bool=False, n_repeat:int = 0, buffer_size:int=1_000_000):

    generator = data_generator(x, y)
    print('Generating...')
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, x.shape[1]), dtype=tf.string),
            tf.TensorSpec(shape=(None, y.shape[1]), dtype=tf.int32)
        )
    )
    print('Mapping...')
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(text_vectorizer(x), tf.int32)[0], y[0]), tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size)


    if n_repeat > 0:
        return train_dataset.cache().repeat(n_repeat).prefetch(tf.data.AUTOTUNE)
    elif n_repeat == -1:
        return train_dataset.cache().repeat().prefetch(tf.data.AUTOTUNE)
    elif n_repeat == 0:
        return train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    