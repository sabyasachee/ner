import sys
sys.path.insert(0, "..")

from utils import data_converter

def calculate_local_features(sentence, word_index):
    features = {}

    word = sentence[word_index][0]
    postag = sentence[word_index][1]
    features["word.lower"] = word.lower()
    features["word.isupper"] = word.isupper()
    features["word.isdigit"] = word.isdigit()
    features["word.istitle"] = word.istitle()
    features["word.suffix.3"] = word[:-3]
    features["word.suffix.2"] = word[:-2]
    features["postag"] = postag
    features["postag.prefix.2"] = postag[:2]
    features["bias"] = True

    if word_index > 0:
        prev_word = sentence[word_index - 1][0]
        prev_postag = sentence[word_index - 1][1]
        features["prev.word.lower"] = prev_word.lower()
        features["prev.word.isupper"] = prev_word.isupper()
        features["prev.word.isdigit"] = prev_word.isdigit()
        features["prev.word.istitle"] = prev_word.istitle()
        features["prev.postag"] = prev_postag
        features["prev.postag.prefix.2"] = prev_postag[:2]
    else:
        features["start"] = True

    if word_index < len(sentence) - 1:
        next_word = sentence[word_index + 1][0]
        next_postag = sentence[word_index + 1][1]
        features["next.word.lower"] = next_word.lower()
        features["next.word.isupper"] = next_word.isupper()
        features["next.word.isdigit"] = next_word.isdigit()
        features["next.word.istitle"] = next_word.istitle()
        features["next.postag"] = next_postag
        features["next.postag.prefix.2"] = next_postag[:2]
    else:
        features["end"] = True

    return features

def add_local_features(sentences):
    local_features = []

    for sentence in sentences:
        sentence_features = []
        for word_index in range(len(sentence)):
            word_features = calculate_local_features(sentence, word_index)
            sentence_features.append(word_features)
        local_features.append(sentence_features)
    
    return local_features
