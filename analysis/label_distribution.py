import sys
sys.path.insert(0, "..")

from utils import data_converter
from collections import Counter

def print_label_distribution(fname):
    sentences = data_converter.read_data(fname)
    labels = data_converter.get_column(sentences, -1)

    flattened_labels = []
    for sentence_label in labels:
        for token_label in sentence_label:
            if token_label != "O":
                token_label = token_label.split("-")[1]
            flattened_labels.append(token_label)
    
    distribution = Counter(flattened_labels).items()
    distribution_tuples = [(entity, number) for entity, number in distribution]
    distribution_tuples = sorted(distribution_tuples, key = lambda t: t[1], reverse = True)
    print("{} samples".format(len(flattened_labels)))
    print("{} unique labels".format(len(distribution_tuples)))
    for entity, number in distribution_tuples:
        print("{:20s} = {:10d}, {:5.2f}%".format(entity, number, 100*(number/len(flattened_labels))))
    print()

def main():
    print("train entity distribution")
    print_label_distribution("../data/onto.train")
    print("test entity distribution")
    print_label_distribution("../data/onto.testa")

if __name__ == "__main__":
    main()
    