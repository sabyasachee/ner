import sys, os
path = os.path.dirname(__file__)
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

def crf_eval(dev_fname, model_file):
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

def crf_infer(test_fname, model_file, out_fname):
    test_sentences = data_converter.read_data(test_fname)
    test_features = local_features.add_local_features(test_sentences)

    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open(model_file)
    test_predictions = [crf_tagger.tag(xseq) for xseq in test_features]

    with open(out_fname, "w") as fw:
        for predictions in test_predictions:
            for prediction in predictions:
                fw.write("{}\n".format(prediction))
            fw.write("\n")
    print("crf output written")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("""usage: python crf.py mode\n\tmode :\n\t\ttrain = train crf and write model file\n\t\teval = evaluate crf model on dev set\n\t\tinfer = infer
                model on test set""")

    model_file = os.path.join(path, "../models/crf.model.bin")
    train_fname = os.path.join(path, "../data/onto.train")
    dev_fname = os.path.join(path, "../data/onto.testa")
    test_fname = os.path.join(path, "../data/onto.testb")
    out_fname = os.path.join(path, "../output/crf_output.txt")
    mode = sys.argv[1]
    
    if mode == "train":
        crf_model(train_fname, model_file)
    elif mode == "eval":
        crf_eval(dev_fname, model_file)
    elif mode == "infer":
        crf_infer(test_fname, model_file, out_fname)
