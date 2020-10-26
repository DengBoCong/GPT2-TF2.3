import os
import jieba
import tensorflow as tf
import model.gpt2 as gpt2
import common.data_utils as data_utils
from config.get_config import get_config


def response(sentence):
    inputs = " ".join(jieba.cut("cls" + sentence + "sep"))
    config = get_config()
    tokenizer = data_utils.load_dict(config["gpt2_dict"])
    inputs = [tokenizer.get(i, 1) for i in inputs.split(' ')]
    # inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=config["max_length"], padding="post")
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.cast(tf.expand_dims(inputs, axis=0), dtype=tf.int64)

    checkpoint_dir = config["checkpoint_dir"]
    model = gpt2.gpt2(vocab_size=config["vocab_size"], num_layers=config["num_layers"],
                      units=config["units"], deep=config["deep"], num_heads=config["num_heads"],
                      dropout=config["dropout"])

    learning_rate = gpt2.CustomSchedule(config["deep"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if os.listdir(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    result = []
    for _ in range(config["max_length"]):
        # print(inputs)
        # exit(0)
        predictions = model(inputs=inputs, training=False)
        predictions = tf.nn.softmax(predictions, axis=-1)
        predictions = predictions[:, -1:, :]
        predictions = tf.squeeze(predictions, axis=1)
        # print(predictions)
        # exit(0)
        pred = tf.argmax(input=predictions, axis=-1)
        print(inputs)
        print(pred)
        # exit(0)
        if pred.numpy()[0] == 2:
            break
        result.append(pred.numpy()[0])

        inputs = tf.concat([inputs, tf.expand_dims(pred, axis=0)], axis=-1)
        print(inputs)
    print(result)
    return data_utils.sequences_to_texts(result, tokenizer)
