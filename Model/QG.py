import tensorflow as tf
import numpy as np

# 加载模型
# model = tf.keras.models.load_model('seq2seq_model')
model = tf.keras.models.load_model('seq2seq_model_val')

# 加载词汇表
# with open('../data/wiki80/vocab.txt', 'r', encoding='utf-8') as f:
with open('../data/wiki80/vocab_val.txt', 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
vocab_size = len(vocab)


# 定义一些帮助函数
def preprocess_input(input_str):
    input_seq = [vocab.index(token) for token in input_str.split()]
    input_seq = np.array(input_seq).reshape(1, -1)
    return input_seq


def generate_question(input_str):
    input_seq = preprocess_input(input_str)
    output_seq = model.predict(input_seq)
    output_seq = np.argmax(output_seq, axis=-1)[0]
    output_str= ' '.join([vocab[token] for token in output_seq])
    return output_str


# 使用模型生成问题
# input_str = 'Merpati flight 106 departed Jakarta on a domestic flight to Tanjung Pandan .'
input_str = 'Vahitahi has a territorial airport .'
output_str = generate_question(input_str)
print('111')
print(output_str)  # 输出：What is the destination of Merpati flight 106?
print('222')
