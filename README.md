# MultiEntity Sentiment Analysis NLP (Lexicon Based and Deep NN based)

## Lexicon Based
```
Prerequistes
```
* wordcloud
* neuralcoref(pip install git+htt!ps://github.com/huggingface/neuralcoref.git)
* nltk
* spacy

```Data
Lexicons from IBM debator, available at https://developer.ibm.com/exchanges/data/all/sentiment-composition-lexicons/
```

```
Steps
```
* Change the input file to your own file, with text to analyze
* Change the lexicon file paths to your system directory path
* Run the MultiEntitySentimentAnalyzer.py to predict 


```
How it works
```

* It is a lexicon based method which uses unigrams, bigrams and polarity reverser lists from 
IBM Debator
* The example document "005.txt" is a political article about UK pre election budget controversy.
Important characters are "Sir Digby", "Mr Brown" and "Mr Balls", with "Mr Brown" capturing most of the focus. Wordcloud correctly captures MrBrown along with other words such as "election"
* In this article "Mr Balls" is positive towards Mr Brown's economic policy, Sir Digby is wary of "Mr Brown". Though the analyzer identifies positive sentiment of Mr Balls towards Mr Brown. It doesn't catch Sir Digby's negative sentiment towards Mr Brown, as it fails to identify Mr Brown as object in the sentence "He was speaking as Sir Digby Jones, CBI director general, warned Mr Brown not to be tempted to use any extra cash on pre-election bribes"

```
Future Enhancements
```
* As the analyzer fails to capture subject, object sometimes, an LDA approach could be explored. Instead of identifying frequent tokens in a document, we could identify concepts(from a fixed set perhaps) and then analyze sentiment of each entity towards each of the concepts.
* Sometimes the analyzer produces noise such as "Mr Brown"-->"rise" = positive(0.344). This does not use the complete object token. It could be either discarded, or could be retained based on a configurable temperature parameter.
* A better subject-object model could be trained, but this would require a huge labelled dataset.

## BERT Based
```
Prerequistes
```
* torch
* transformers
* pickle
* nltk
* glob2
* bs4

```
Steps
```
* Follow the steps in subject-object_MPQA.ipynb for your own data preparation
This prepares a tagged sequence for subject-object identification in a sentence.
It follows BIO format
* Follow steps in Subject-Object-BERT-SEQ.ipynb to train your own model
the encoder is BERT layer followed by a GRU decoder
* Model to be released soon [training underway]
* If above model is successful, a classifier model to predict polarity-one of the following classes from the corpus-
positive, negative, both, neutral,uncertain-positive, uncertain-negative, uncertain-both,
                uncertain-neutral
```
Limitations

The MPQA corpus has only 70 documents with around 10 sentences each. This may not be enough, even with a pretrained model.
```

## Any questions 👨‍💻
<p> If you have any questions, feel free to ask me: </p>
<p> 📧: "nirmalendu@outlook.com"</p>