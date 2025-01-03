from collections import defaultdict
from functools import reduce
from typing import List, Tuple, Dict
import math
import re
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


with open("data.txt", "r", errors="ignore", encoding="utf-8") as f:
    paragraph = f.read()
    f.close()


class W2W:
    """Class W2W implementing Word2Vec Skipgram model"""

    def __init__(self, text: str) -> None:
        self.text: str = text
        self.window_size: int = 2
        self.dimension: int = 8
        self.learning_rate: float = 0.01
        self.epochs: int = 1000
        self.vocab_size, self.words, self.word_index, self.index_word, self.corpus = (
            self._tokenizer()
        )
        self.train_data: List[Tuple[List[int], List[int]]] = (
            self._generate_target_context_pairs()
        )
        self.w1: List[List[float]] = np.random.rand(self.vocab_size, self.dimension)
        self.w2: List[List[float]] = np.random.rand(self.dimension, self.vocab_size)

    def _tokenizer(
        self,
    ) -> Tuple[int, List[str], Dict[str, int], Dict[int, str], List[List[str]]]:
        """Tokenizing the text to tokens"""
        nltk.download("stopwords")
        nltk.download("wordnet")
        wnl = WordNetLemmatizer()
        stop_words = stopwords.words("english")
        rep_characters = [
            ".",
            ",",
            "'",
            '"',
            "-",
            "(",
            ")",
            "-",
            "@",
            "!",
            "$",
            "#",
            "*",
            "$",
            "&",
            "^",
            "~",
            "`",
            "[",
            "]",
        ]
        clean_text = re.sub(r"[\d+]", "", self.text)

        def _sanitizer(sentence):
            return reduce(
                lambda sentence, rep_characters: sentence.replace(rep_characters, " "),
                rep_characters,
                sentence,
            )

        x = list(
            map(lambda sentence: _sanitizer(sentence), clean_text.lower().split("."))
        )
        corpus = []
        for i in x:
            corpus.append([wnl.lemmatize(y) for y in i.split() if y not in stop_words])

        corpus = corpus[:-1]
        words = []
        for i in corpus:
            words += i
        dictionary = defaultdict(int)
        for word in words:
            dictionary[word] += 1
        vocab_size = len(dictionary.keys())
        words = list(dictionary.keys())
        word_index = {word: index for index, word in enumerate(words)}
        index_word = {index: word for index, word in enumerate(words)}

        return vocab_size, words, word_index, index_word, corpus

    def _generate_target_context_pairs(self) -> List[Tuple[List[int], List[int]]]:
        entire_corpus = []
        e_c = []
        for sentence in self.corpus:
            sentence_length = len(sentence)
            target_context_pairs = []
            t_c_p = []
            for i, word in enumerate(sentence):
                target_word = self._one_hot_encoded(word)
                t_g = word
                context_word = []
                c = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and j <= sentence_length - 1 and j >= 0:
                        context_word.append(self._one_hot_encoded(sentence[j]))
                        c.append(sentence[j])
                t_c_p.append([t_g, c])
                target_context_pairs.append([target_word, context_word])
            entire_corpus.append(target_context_pairs)
            e_c.append(t_c_p)
        return entire_corpus

    def _one_hot_encoded(self, word: str) -> List[int]:
        one_hot_vector = [0] * self.vocab_size
        index = self.word_index[word]
        one_hot_vector[index] = 1
        return one_hot_vector

    def train(self) -> None:
        """Training the model by using target-context pairs"""
        writer = SummaryWriter("runs/w2loss")
        for i in tqdm(range(self.epochs)):
            epoch_loss = 0
            for x in self.train_data:
                for target, context_words in x:
                    loss = 0
                    target_embedding, prediction = self._forward_pass(target)
                    EI = np.sum(
                        [np.subtract(prediction, word) for word in context_words],
                        axis=0,
                    )
                    self._backward_prop(EI, target_embedding, target)
                    for word in context_words:
                        loss += self._cross_entropy_loss(prediction, word)
                    epoch_loss += loss
            writer.add_scalar("Training Loss", epoch_loss, i)

    def _forward_pass(self, target: List[int]) -> Tuple[List[int], List[float]]:
        target_embedding = np.dot(target, self.w1)
        context_embedding = np.dot(target_embedding, self.w2)
        prediction = self._softmax(context_embedding)
        return target_embedding, prediction

    def _cross_entropy_loss(self, pred: List[float], actual: List[int]) -> float:
        loss = 0
        for i, predict in enumerate(pred):
            loss += -1 * actual[i] * np.log(predict)
        return loss

    def _backward_prop(self, e_i, target_embedding, target) -> None:
        # FINDING GRADIENTS
        dw_d1 = np.outer(target_embedding, e_i)
        dw_d2 = np.outer(target, np.dot(self.w2, e_i.T))

        # UPDATING WEIGHTS
        self.w1 = self.w1 - (self.learning_rate * dw_d2)
        self.w2 = self.w2 - (self.learning_rate * dw_d1)

    def _softmax(self, x: List[float]) -> List[float]:
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = np.dot(a, b)
        product_of_length = math.sqrt(sum([i**2 for i in a])) * math.sqrt(
            sum(i**2 for i in b)
        )
        return dot_product / product_of_length

    def similiar_words(self, word: str) -> List[str]:
        """Finding similiar words using cosine similiarity"""
        cosine_similarities = []
        for i in range(self.vocab_size):
            cosine_similarities.append(
                [
                    self._cosine_similarity(self.w1[self.word_index[word]], self.w1[i]),
                    self.index_word[i],
                ]
            )
        cosine_similarities.sort()
        similiar = []
        for i in cosine_similarities[-10:]:
            similiar.append(i[1])
        return similiar[::-1][1:]

    def save_model(self):
        """Saving the Embedding Matrix"""
        np.save("embedding.npy", self.w1)

    def load_model(self):
        """Loading the saved Embedding Matrix"""
        self.w1 = np.load("embedding.npy")
        return self.w1


def main():
    """Training and Saving the Model"""
    w2v = W2W(paragraph)
    w2v.train()
    w2v.save_model()


if __name__ == "__main__":
    main()
