# MultiEntity Sentiment Analysis NLP

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


## Any questions 👨‍💻
<p> If you have any questions, feel free to ask me: </p>
<p> 📧: "nirmalendu@outlook.com"></p>