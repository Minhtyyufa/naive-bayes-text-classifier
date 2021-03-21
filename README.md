# Naive Bayes Text Classifier

This is my implementation of a Naive Bayes text classifier that I made for a natural language processing course. 

## To run the classifier

Install dependencies
```buildoutcfg
pip install -r requirements.txt
```

Run the program
```buildoutcfg
python3 naive_bayes_text_classifier.py
```

## Dependencies
- [Natural Language Toolkit](https://www.nltk.org/)
- [NumPy](https://numpy.org/)

## About my classifier
The classifier uses the traditional Naive Bayes method. It uses a RegEx tokenizer that only tokenizes words and ignores all punctuation. The model is adaptable and can be easily changed to use a different tokenizer if need be. No weighting scheme is used for the tokens. The model uses Laplace smoothing with an alpha value of .25.

The model can take in several parameters including: tokenizer, which allows you to choose either the nltk tokenizer or a tokenizer that only tokenizes words; stopwords, which allows you to choose whether stopwords are ignored; stemmer, which specifies whether a stemmer is used; lemmatizer, which specifies whether a lemmatizer is used; smoothing, which specifies what alpha value is used for Laplace smoothing; and lower-case, which changes all of the tokens to lowercase.    

To choose the default parameters for the model, I swept across all combinations of the optional parameters, as seen in the sweep_params function. For each set of parameters, the model is trained and tested from scratch 10 times, each time randomly scrambling which documents are part of the test set and training set. I then used the model parameters that produced the highest average accuracy. 

For the datasets that only provided the training documents, I randomly split them into an 80-20 training-test set.
  