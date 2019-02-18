# Output

The predictions for onto.testb are in output folder. The filenames are crf_output.txt and lstm_output.txt. The report is the file report.pdf

# Installation

Ensure python 3 environment. You'll need to pip install the following

```
pip install python-crfsuite
pip install flair
mkdir models
```

# Usage

## CRF

```
cd training
python crf.py mode
```
**mode** can take the values *train*, *eval* and *infer*. *train* trains the crf model and saves it under models directory. *eval* prints the development set performance. *infer* writes the predictions of test set in output directory.

## LSTM
```
cd training
python lstm.py mode
```
**mode** can take the values *train* and *infer*. *train* trains the crf lstm model, saves it under models directory and prints the development set performance. *infer* writes the predictions of test set in output directory.

Please note, training the crf lstm model takes atleast 4 hours. Please ensure access to a GPU. Inference takes atleast ten minutes.