import sys, os
sys.path.append("..")
path = os.path.dirname(__file__)

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List

def train():
    columns = {0: 'word', 1: 'postag', 2: 'ner'}
    data_folder = os.path.join(path, "../data/")

    corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns, train_file = "onto.train", dev_file = "onto.testa", test_file="onto.testa")

    print(corpus)

    tag_dictionary = corpus.make_tag_dictionary(tag_type = "ner")
    print(tag_dictionary.idx2item)

    embedding_types: List[TokenEmbeddings] = [WordEmbeddings("glove"), CharacterEmbeddings()]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size = 256, embeddings = embeddings, tag_dictionary = tag_dictionary, tag_type = "ner", use_crf = True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    model_path = os.path.join(path, "../models/")

    trainer.train(model_path, learning_rate = 0.1, mini_batch_size = 64, max_epochs = 5)

if __name__ == "__main__":
    train()
    test()