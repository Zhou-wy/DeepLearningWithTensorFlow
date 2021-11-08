import random
import numpy as np


class ControlData:
    def __init__(self):
        self.data = []
        with open("data/data.txt", mode = "r", encoding = "utf-8") as fr:
            lines = fr.readlines()[0]
            lines = lines.strip().split(', ')
            for idx in range(len(lines)):
                self.data.append(float(lines[idx]))
            # print(self.data)

    def load_data(self):
        idx_to_char = list(set(self.data))  # set() 函数创建一个无序不重复元素集
        # print("idx_to_char", idx_to_char)
        char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
        # print("char_to_idx", char_to_idx)
        vocab_size = len(char_to_idx)
        corpus_indices = [char_to_idx[char] for char in self.data]
        """
        print("corpus_indices:", corpus_indices, "\n", "char_to_idx:", char_to_idx, "\n", "idx_to_char:", idx_to_char,
              "\n", "vocab_size:", vocab_size)
        """
        return corpus_indices, char_to_idx, idx_to_char, vocab_size

    def data_iter_random(self, corpus_indices, batch_size, num_steps, ctx = None):
        # 因为输出的索引是相应输入的索引加1
        num_examples = (len(corpus_indices) - 1) // num_steps
        epoch_size = num_examples // batch_size
        example_indices = list(range(num_examples))
        random.shuffle(example_indices)

        # 返回从pos开始的长为num_steps的序列
        def _data(pos):
            return corpus_indices[pos: pos + num_steps]

        for i in range(epoch_size):
            # 每次读取batch_size个随机样本
            i = i * batch_size
            batch_indices = example_indices[i: i + batch_size]
            X = [_data(j * num_steps) for j in batch_indices]
            Y = [_data(j * num_steps + 1) for j in batch_indices]
            yield np.array(X, ctx), np.array(Y, ctx)

    def data_iter_consecutive(self, corpus_indices, batch_size, num_steps, ctx = None):
        corpus_indices = np.array(corpus_indices)
        data_len = len(corpus_indices)
        batch_len = data_len // batch_size
        indices = corpus_indices[0: batch_size * batch_len].reshape((
            batch_size, batch_len))
        epoch_size = (batch_len - 1) // num_steps
        for i in range(epoch_size):
            i = i * num_steps
            X = indices[:, i: i + num_steps]
            Y = indices[:, i + 1: i + num_steps + 1]
            yield X, Y

    def TureData(self):
        return self.data


"""d = ControlData()
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d.load_data()
batch_size, num_steps = 16, 18
data_iter = d.data_iter_consecutive(corpus_indices, batch_size, num_steps)
for X,Y in data_iter:
    print(X,Y)"""
