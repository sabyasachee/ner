import sys
sys.path.append("..")

import pycrfsuite
from utils import data_converter, conlleval
from features import local_features

def crf_model(train_fname, model_file):
    train_sentences = data_converter.read_data(train_fname)
    train_features = local_features.add_local_features(train_sentences)
    train_labels = data_converter.get_column(train_sentences, -1)

    crf_trainer = pycrfsuite.Trainer(verbose = True)
    crf_trainer.set_params({"c1": 0.1, "c2": 0.01, "max_iterations": 200, "feature.possible_transitions": True})
    for xseq, yseq in zip(train_features, train_labels):
        crf_trainer.append(xseq, yseq)
    
    crf_trainer.train(model_file)

def crf_infer(dev_fname, model_file):
    dev_sentences = data_converter.read_data(dev_fname)
    dev_features = local_features.add_local_features(dev_sentences)
    dev_labels = data_converter.get_column(dev_sentences, -1)

    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open(model_file)
    dev_predictions = [crf_tagger.tag(xseq) for xseq in dev_features]
    
    iterable = []
    for labels, predictions in zip(dev_labels, dev_predictions):
        for label, prediction in zip(labels, predictions):
            iterable.append("dummy\t{}\t{}".format(label, prediction))
        iterable.append("")
    conlleval.evaluate(iterable)

if __name__ == "__main__":
    # crf_model("../data/onto.train","../models/crf.model.bin")
    crf_infer("../data/onto.testa","../models/crf.model.bin")