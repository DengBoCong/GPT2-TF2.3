import os
import json
import jieba
import tensorflow as tf
from config.get_config import get_config


def preprocess_raw_data(raw_data, tokenized_data):
    if not os.path.exists(raw_data):
        print("数据集不存在，请添加数据集")
        exit(0)

    pairs = []
    count = 0
    config = get_config()

    with open(raw_data, encoding="utf-8") as file:
        pair = ""
        for line in file:
            line = line.strip("\n").replace('/', '')

            if line == "":
                pairs.append(pair)
                count += 1
                if count % 10000 == 0:
                    print('已读取：', count, '轮问答对')
                pair = ""
                continue
            elif len(pair) == 0:
                pair = config["cls_token"] + line + config["sep_token"]
            else:
                pair = pair + line + config["sep_token"]

    print("数据读取完毕，正在处理中...")

    with open(tokenized_data, 'w', encoding="utf-8") as file:
        for i in range(len(pairs)):
            file.write(" ".join(jieba.cut(pairs[i])) + "\n")
            if i % 10000 == 0:
                print(len(range(len(pairs))), '处理进度：', i)


def creat_padding_mask(inputs):
    """
    对input中的padding单位进行mask
    :param inputs: 句子序列输入
    :return: 填充部分标记
    """
    mask = tf.cast(tf.math.equal(inputs, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def creat_look_ahead_mask(inputs):
    sequence_length = tf.shape(inputs)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
    padding_mask = creat_padding_mask(inputs)
    return tf.maximum(look_ahead_mask, padding_mask)


def load_dict(dict_fn):
    with open(dict_fn, encoding="utf-8") as file:
        token = json.load(file)
    return token


def sequences_to_texts(sequences, token_dict):
    """
    将序列转换成text
    """
    inv = {}
    for key, value in token_dict.items():
        inv[value] = key

    result = ""
    for token in sequences:
        result += inv[token]
    return result


def load_data(tokenized_data, dict_fn, num_sample=0):
    if not os.path.exists(tokenized_data):
        print("没有检测到分词数据集，请先执行pre_treat模式")
        exit(0)

    with open(tokenized_data, 'r', encoding="utf-8") as file:
        lines = file.read().strip().split("\n")

        if num_sample == 0:
            sentences = [line for line in lines]
        else:
            sentences = [line for line in lines[:num_sample]]

    config = get_config()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=config["unk_token"])
    tokenizer.fit_on_texts(sentences)
    input_tensor = tokenizer.texts_to_sequences(sentences)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=config["max_length"],
                                                                 padding="post")

    with open(dict_fn, "w", encoding="utf-8") as file:
        file.write(json.dumps(tokenizer.word_index, indent=4, ensure_ascii=False))

    dataset = tf.data.Dataset.from_tensor_slices(input_tensor).cache().shuffle(
        config["buffer_size"]).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(config["batch_size"], drop_remainder=True)

    steps_per_epoch = len(input_tensor) // config["batch_size"]

    return dataset, tokenizer, steps_per_epoch


if __name__ == '__main__':
    # mask = creat_look_ahead_mask(tf.constant([[1, 2, 3, 0, 0, 0], [1, 2, 3, 1, 2, 0], [1, 2, 3, 1, 0, 0]]))
    # print(mask)

    load_data("data/tokenized_data.txt", "data/gpt2_dict.json")
