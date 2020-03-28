#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:24:21 2020

@author: nirmalenduprakash
Identifies important entities in a document and discovers sentiment towards these 
entities
"""

import spacy
import numpy as np
import pandas as pd

#!pip install git+htt!ps://github.com/huggingface/neuralcoref.git
import nltk
#nltk.download('punkt')
import neuralcoref
from wordcloud import WordCloud, STOPWORDS

nlp = spacy.load("en_core_web_sm")
stopwords = set(STOPWORDS)
pronouns=['i','he','she','you','we','they','them','it','his','who']
 

def read_file(filepath):
  df=pd.read_table(filepath,sep=' ')
  if('POS_TAGS' in df.columns):
    df.drop(['POS_TAGS'],inplace=True,axis=1)
  return df 

df_ug=read_file('/Users/nirmalenduprakash/Documents/Project/NLP/Sentiment Mining/IBM_Debater_(R)_SC_COLING_2018/LEXICON_UG.txt')
df_bg=read_file('/Users/nirmalenduprakash/Documents/Project/NLP/Sentiment Mining/IBM_Debater_(R)_SC_COLING_2018/LEXICON_BG.txt')
reversers=pd.read_excel\
    ('/Users/nirmalenduprakash/Documents/Project/NLP/Sentiment Mining/IBM_Debater_(R)_SC_COLING_2018/SEMANTIC_CLASSES.xlsx'\
     ,sheet_name=['REVERSER_POS','REVERSER_NEG'])
pos_reversers=list(reversers['REVERSER_POS'].items())
pos_reversers=list(pos_reversers[0][1])
pos_reversers.append(reversers['REVERSER_POS'].keys()[0])
# pos_reversers[0][0].extend()
neg_reversers=list(reversers['REVERSER_NEG'].items())
neg_reversers=list(neg_reversers[0][1])
neg_reversers.append(reversers['REVERSER_NEG'].keys()[0])


def polarity(document,imptwords=None):
  nlp = spacy.load("en_core_web_sm")
  neuralcoref.add_to_pipe(nlp)
  doc=nlp(document)
  dict_ents={}
  for ent in doc.ents:
    if(ent.label_ in ['ORG','PERSON','NORP']):
      dict_ents[ent.text]=ent.label_
  result=[]
  for sent in nltk.sent_tokenize(doc._.coref_resolved):
    if(imptwords is None or sum([1 for word in imptwords if word in str(sent)])>0):
      doc=nlp(str(sent))
      chunks=[]
      for chunk in doc.noun_chunks:
        chunks.append((chunk.text, chunk.root.dep_,chunk.root.head.text))
      sent_subj=None
      sent_obj=None
      polarity=0
      for chunk in reversed(chunks):     
        if(chunk[1]in ['dobj']):
          sent_obj=chunk[0]
        elif(chunk[1]in ['nsubj']):
          sent_subj=chunk[0]
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

def bigram_compare(w1,w2):
  w1_bigrams=list(nltk.bigrams(w1.split()))
  w2_bigrams=list(nltk.bigrams(w2.split()))
  if(len(w1_bigrams)==0 or len(w2_bigrams)==0):
    return compare(w1,w2)
  for tok1,tok2 in w1_bigrams:
    for t1,t2 in w2_bigrams:
      if(tok1==t1 and tok2==t2):
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

def document_polarity(doc):
    # doc=read_file_content('/content/drive/My Drive/005.txt')
    wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=5,
            max_font_size=40, 
            scale=3,
            random_state=1
        ).generate(str(doc))
    imptwords=[k for k,v in wordcloud.words_.items()]   

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
    return imptwords,refined



    
if  __name__== "__main__":       
    #Score Calculation        
    file_names=list(data['file_name'].unique())
    unigrams=[]
    bigrams=[]
    for index,doc in enumerate(documents):
      imptwords,sent_doc=document_polarity(doc)
      sent_doc_gt=data[data['file_name']==file_names[index]]
      print(imptwords)
      print(sent_doc)
      refined=[]
      for indx,row in sent_doc_gt.iterrows():
        if((sum([1 for w2 in imptwords if(compare(row['agent'],w2))])>0 or sum([1 for w2 in imptwords if(compare(row['target'],w2))])>0)
              and (row['agent'].lower().strip() not in pronouns) and (row['target'].lower().strip() not in pronouns)
              and ('pos' in row['sentiment_type'] or 'neg' in row['sentiment_type'])):
              refined.append((row['agent'],row['target'],row['sentiment_type'],row['sentiment_intensity']))
      # Dircard findings where agent is same as target
      # 1 token match between agent and target of predicted and ground truth
      # score is matched polarity items/total items in ground truth
      print(refined)
      match=0
      for pred_agent,pred_target,score in sent_doc:
        for agent,target,sent_type,sent_intensity in refined:
          if(pred_agent.strip()!=pred_target.strip() and compare(agent,pred_agent)
            and compare(target,pred_target) and ((score>0 and 'pos' in sent_type) or (score<0 and 'neg' in sent_type))):
            match+=1
            break
      bigram_match=0      
      for pred_agent,pred_target,score in sent_doc:
        for agent,target,sent_type,sent_intensity in refined:
          if(pred_agent.strip()!=pred_target.strip() and bigram_compare(agent,pred_agent)
            and bigram_compare(target,pred_target) and ((score>0 and 'pos' in sent_type) or (score<0 and 'neg' in sent_type))):
            bigram_match+=1
            break 
      if(len(refined)>0):
        unigrams.append(match/len(refined)) 
        bigrams.append(bigram_match/len(refined))           
        print('unigram match {}'.format(match/len(refined)))
        print('bigram match {}'.format(bigram_match/len(refined)))     
      print('============================================================================')
    print('average unigram score: {}'.format(np.mean(unigrams)))
    print('average bigram score: {}'.format(np.mean(bigrams)))