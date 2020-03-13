#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:24:21 2020

@author: nirmalenduprakash
Identifies important entities in a document and discovers sentiment towards these 
entities
"""

import spacy
import pandas as pd
#!pip install git+htt!ps://github.com/huggingface/neuralcoref.git
import nltk
nltk.download('punkt')
import neuralcoref
from wordcloud import WordCloud, STOPWORDS


def read_file(filepath):
  df=pd.read_table(filepath,sep=' ')
  if('POS_TAGS' in df.columns):
    df.drop(['POS_TAGS'],inplace=True,axis=1)
  return df 

def polarity(document,imptwords):
  nlp = spacy.load("en_core_web_sm")
  neuralcoref.add_to_pipe(nlp)
  doc=nlp(document)
  dict_ents={}
  for ent in doc.ents:
    if(ent.label_ in ['ORG','PERSON','NORP']):
      dict_ents[ent.text]=ent.label_
  result=[]
  # print(dict_ents)
  
  for sent in nltk.sent_tokenize(doc._.coref_resolved):
    if(sum([1 for word in imptwords if word in str(sent)])>0):
      doc=nlp(str(sent))
      chunks=[]
      for chunk in doc.noun_chunks:
        # print(chunk)
        chunks.append((chunk.text, chunk.root.dep_,chunk.root.head.text))
      sent_subj=None
      sent_obj=None
      polarity=0
      for chunk in reversed(chunks):
        # key=[key for key in dict_ents.keys() if key in chunk[0]]
        if(chunk[1]in ['dobj']):
          sent_obj=chunk[0]
        elif(chunk[1]in ['nsubj']):
          sent_subj=chunk[0]
      # bigrams=list(nltk.bigrams(reversed(str(sent).split())))
      sent_tokenized=list(reversed(sent.strip('.').split()))
      bigrams=[]
      for i in range(len(sent_tokenized)):
        try:
          bigrams.append((sent_tokenized[i],sent_tokenized[i+1]))
          i+=2
        except:
          break 
      for t1,t2 in bigrams:
        t1=t1.replace('\"','')
        t2=t2.replace('\"','')
        itm=df_bg[df_bg['BIGRAM']==str(t1)+'-'+str(t2)]
        if len(itm)>0:
          polarity+=float(itm['SENTIMENT_SCORE'])
        elif(t2 in pos_reversers):
          itm=df_ug[df_ug['UNIGRAM']==str(t1)]          
          if len(itm)>0:
            val=float(itm['SENTIMENT_SCORE'])
            if(val<0):
              polarity+= -1*float(itm['SENTIMENT_SCORE'])
        elif(t1 in neg_reversers):
          itm=df_ug[df_ug['UNIGRAM']==str(t1)]          
          if len(itm)>0:
            val=float(itm['SENTIMENT_SCORE'])
            if(val>0):
              polarity+= -1*float(itm['SENTIMENT_SCORE'])      
        else:
          itm=df_ug[df_ug['UNIGRAM']==str(t1)]          
          if len(itm)>0:
            polarity+= float(itm['SENTIMENT_SCORE'])
          itm=df_ug[df_ug['UNIGRAM']==str(t2)]
          if len(itm)>0:
            polarity+= float(itm['SENTIMENT_SCORE'])
      result.append((sent_subj,sent_obj,polarity)) 
  return result,dict_ents

def read_file_content(path):
  with open(path,'r') as f:
    text=f.read()
  return text 

def compare(w1,w2):
  for w in w1.split():
    if(w in w2.split()):
      return True
  return False 

def getImportantWords(doc):
  wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=5,
            max_font_size=40, 
            scale=3,
            random_state=1
        ).generate(str(doc))
    imptwords=[k for k,v in wordcloud.words_.items()]
    return imptwords     
    
if  __name__== "__main__:
    df_ug=read_file('/content/drive/My Drive/IBM_Debater_(R)_SC_COLING_2018/LEXICON_UG.txt')
    df_bg=read_file('/content/drive/My Drive/IBM_Debater_(R)_SC_COLING_2018/LEXICON_BG.txt')
    reversers=pd.read_excel\
        ('/content/drive/My Drive/IBM_Debater_(R)_SC_COLING_2018/SEMANTIC_CLASSES.xlsx'\
         ,sheet_name=['REVERSER_POS','REVERSER_NEG'])
    pos_reversers=list(reversers['REVERSER_POS'].items())
    pos_reversers=list(pos_reversers[0][1])
    pos_reversers.append(reversers['REVERSER_POS'].keys()[0])
    # pos_reversers[0][0].extend()
    neg_reversers=list(reversers['REVERSER_NEG'].items())
    neg_reversers=list(neg_reversers[0][1])
    neg_reversers.append(reversers['REVERSER_NEG'].keys()[0])    
        
    stopwords = set(STOPWORDS)
    pronouns=['i','he','she','you','we','they','them','it','his']
    doc=read_file_content('/content/drive/My Drive/bbc/politics/005.txt')
    imptwords=getImportantWords(doc) 
    scores,dict_ents=polarity(doc,imptwords)
    refined=[]
    for subj,obj,score in scores:
      if(subj is not None and obj is not None):#and sum([1 for key in dict_ents.keys() if (compare(subj,key) and compare(obj,key))])>0):
        if((sum([1 for w2 in imptwords if(compare(subj,w2))])>0 or sum([1 for w2 in imptwords if(compare(obj,w2))])>0)
          and (subj.lower() not in pronouns) and (obj.lower() not in pronouns)):
          refined.append((subj,obj,score))
        elif(subj is None and obj is not None and obj.lower() not in pronouns and sum([1 for w2 in imptwords if(compare(obj,w2))])>0):
          subj='author'
          refined.append((subj,obj,score))           
    print(refined)
