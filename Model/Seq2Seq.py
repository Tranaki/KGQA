import tensorflow as tf
import numpy as np

# 加载数据
# with open('../data/wiki80/wiki80_train_processed.txt', 'r') as f:
with open('../data/wiki80/wiki80_val_processed.txt', 'r') as f:
        data = f.read().splitlines()

# 将数据分为输入序列和输出序列
inputs = []
outputs = []
for line in data:
    input_seq, output_seq = line.split('\t')
    inputs.append(input_seq)
    outputs.append(output_seq)

# 构建词汇表
vocab = set()
for seq in inputs + outputs:
    for token in seq.split():
        vocab.add(token)
vocab_size = len(vocab)

# 将序列转换为数字ID序列
input_seqs = []
output_seqs = []
for input_seq, output_seq in zip(inputs, outputs):
    input_seq = [int(token) for token in input_seq.split()]
    output_seq = [int(token) for token in output_seq.split()]
    input_seqs.append(input_seq)
    output_seqs.append(output_seq)

# 将序列填充为相同的长度
max_len = max(len(seq) for seq in input_seqs + output_seqs)
input_seqs = tf.keras.preprocessing.sequence.pad_sequences(input_seqs, maxlen=max_len, padding='post')
output_seqs = tf.keras.preprocessing.sequence.pad_sequences(output_seqs, maxlen=max_len, padding='post')

# 划分训练集和测试集
split_index = int(len(data) * 0.8)
train_inputs = input_seqs[:split_index]
train_outputs = output_seqs[:split_index]
test_inputs = input_seqs[split_index:]
test_outputs = output_seqs[split_index:]

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.RepeatVector(max_len),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_inputs, train_outputs, validation_data=(test_inputs, test_outputs), epochs=10)

# 评估模型
model.evaluate(test_inputs, test_outputs)

# 保存模型
# model.save('seq2seq_model')
model.save('seq2seq_model_val')
