import os
import sys
import time
import tensorflow as tf
from config.get_config import get_config
import common.data_utils as data_utils
import model.gpt2 as gpt2


def train():
    print('训练开始，正在准备数据中...')

    config = get_config()
    dataset, tokenizer, steps_per_epoch = data_utils.load_data(tokenized_data=config["tokenized_corpus"],
                                                               dict_fn=config["gpt2_dict"], num_sample=35)
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    learning_rate = gpt2.CustomSchedule(config["deep"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model = gpt2.gpt2(vocab_size=config["vocab_size"], num_layers=config["num_layers"],
                      units=config["units"], deep=config["deep"], num_heads=config["num_heads"],
                      dropout=config["dropout"])

    checkpoint_dir = config["checkpoint_dir"]
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if os.listdir(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    for epoch in range(config["epochs"]):
        print('Epoch {}/{}'.format(epoch + 1, config["epochs"]))
        start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        step_loss = 0
        batch_sum = 0
        sample_sum = 0

        for batch, inputs in enumerate(dataset.take(steps_per_epoch)):
            inputs_real = inputs[:, 1:]
            inputs = inputs[:, :-1]
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = loss_function(inputs_real, predictions)
            gradient = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            train_loss(loss)
            train_accuracy(inputs_real, predictions)
            step_loss = train_loss.result()
            batch_sum = batch_sum + len(inputs)
            sample_sum = steps_per_epoch * len(inputs)
            print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                  flush=True)
        step_time = (time.time() - start_time)
        sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                         .format(step_time, step_loss))
        sys.stdout.flush()
        checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))
    print("训练结束")


def loss_function(real, pred):
    config = get_config()
    real = tf.reshape(real, shape=(-1, config["max_length"] - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")(real, pred)
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)

# def loss_function(target, logits):
#     padding = 0
#     mask = tf.math.logical_not(tf.math.equal(target, padding))
#     loss_ = tf.keras.losses.SparseCategoricalCrossentropy(target, logits)
#
#     mask = tf.cast(mask, dtype=tf.float32)
#     loss_ *= mask
#
#     return tf.reduce_mean(loss_)
