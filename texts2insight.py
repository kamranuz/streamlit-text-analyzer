import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import spacy
from spacy.matcher import Matcher
from spacy.attrs import ORTH
from spacy.attrs import LENGTH
from spacy.attrs import LEMMA
from spacy.attrs import POS
from spacy.attrs import IS_PUNCT
from spacy.attrs import IS_SPACE
from collections import Counter


import pymorphy3



class NLP():
    def __init__(self, text_collections, model_name, max_length=40_000_000) -> None:
        self.corpus = dict()
        self.statistics = dict()
        self.statistics_ = dict()
        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        process =  spacy.load(model_name, disable=['ner'])
        process.max_length = max_length


        for i, collection in enumerate(text_collections):
            self.corpus[i]     = spacy.tokens.Doc.from_docs([process(text.getvalue().decode('utf-8')) for text in collection])
            is_punct = (self.corpus[i].to_array([IS_PUNCT])==1) | (self.corpus[i].to_array([IS_SPACE])==1)
            

            self.statistics_[i] = {
                                  'Lemm_all_freq_'       : {process.vocab[lemma].text: count  for lemma,count in self.corpus[i].count_by(LEMMA).items()},

                                  'Омонимонимы_'         :  Counter(word.text for word in self.corpus[i] if self._is_omonim(word.text)),

                                  'Несловарные слова_'   : Counter(i.text for i in self._get_oov(self.corpus[i])),
                
                                  'pos_freq_'            : {process.vocab[pos].text: count  for pos,count in self.corpus[i].count_by(POS).items()},

                                  'pron_case_freq_'      : Counter(' '.join(j.morph.get('Case')) for j in self.corpus[i] if j.pos_=='PRON'),
                                  'pron_gender_freq_'    : Counter(' '.join(j.morph.get('Gender')) for j in self.corpus[i] if j.pos_=='PRON'),
                                  'pron_number_freq_'    : Counter(' '.join(j.morph.get('Number')) for j in self.corpus[i] if j.pos_=='PRON'),
                                  
                                  'noun_case_freq_'      : Counter(' '.join(j.morph.get('Case')) for j in self.corpus[i] if j.pos_=='NOUN'),
                                  'noun_gender_freq_'    : Counter(' '.join(j.morph.get('Gender')) for j in self.corpus[i] if j.pos_=='NOUN'),
                                  'noun_animacy_freq_'   : Counter(' '.join(j.morph.get('Animacy')) for j in self.corpus[i] if j.pos_=='NOUN'),
                                  'noun_number_freq_'    : Counter(' '.join(j.morph.get('Number')) for j in self.corpus[i] if j.pos_=='NOUN'),

                                  'verb_tense_freq_'     : Counter(' '.join(j.morph.get('Tense')) for j in self.corpus[i] if j.pos_=='VERB'),
                                  'verb_gender_freq_'    : Counter(' '.join(j.morph.get('Gender')) for j in self.corpus[i] if j.pos_=='VERB'),

                                  'aux_tense_freq_'      : Counter(' '.join(j.morph.get('Tense')) for j in self.corpus[i] if j.pos_=='AUX'),
                                  'aux_gender_freq_'     : Counter(' '.join(j.morph.get('Gender')) for j in self.corpus[i] if j.pos_=='AUX'),

                                  'adj_case_freq_'       : Counter(' '.join(j.morph.get('Case')) for j in self.corpus[i] if j.pos_=='ADJ'),
                                  'adj_gender_freq_'     : Counter(' '.join(j.morph.get('Gender')) for j in self.corpus[i] if j.pos_=='ADJ'),
                                  'adj_number_freq_'     : Counter(' '.join(j.morph.get('Number')) for j in self.corpus[i] if j.pos_=='ADJ'),
                                  'adj_degree_freq_'     : Counter(' '.join(j.morph.get('Degree')) for j in self.corpus[i] if j.pos_=='ADJ'),

                                  'adv_degree_freq_'     : Counter(' '.join(j.morph.get('Degree')) for j in self.corpus[i] if j.pos_=='ADV'),

                                #   '# Underfined guys'                 : list(j.text for j in self.corpus[i] if j.pos_=='AUX' and len(j.morph.get('Tense'))==0),
                                  } 
            self.statistics[i] = {
                                  '# Cловоупотреблений'         : len(self.corpus[i])-sum(is_punct),
                                  '# Cловоформ'                 : len(set(self.corpus[i].to_array([ORTH])[~is_punct])),
                                  '# Лемм'                      : len(set(self.corpus[i].to_array([LEMMA])[~is_punct])),
                                  '# Несловарных слов'          : len(self.statistics_[i]['Несловарные слова_']),
                                  '# Предложений'               : len(tuple(self.corpus[i].sents)),
                                  'E[Длина предложения]'        : np.array([len(sent) for  sent in self.corpus[i].sents]).mean(),
                                  'E[Длина словоупотребления]'  : self.corpus[i].to_array([LENGTH])[~is_punct].mean(),
                                  '# Омонимов'                  : len(self.statistics_[i]['Омонимонимы_']),
                                  'E[Длина словоупотребления]'  : self.corpus[i].to_array([LENGTH])[~is_punct].mean(),
                                  } 
            self.statistics[i]['К. Лексического богатства'] =  self.statistics[i]['# Лемм']/self.statistics[i]['# Cловоформ']
            self.statistics[i]['% Омонимов']                =  self.statistics[i]['# Омонимов'] / self.statistics[i]['# Cловоформ']

    def make_stats_table(self):
        df = pd.DataFrame(self.statistics)
        df.columns = [f'Коллеция №{i}' for i in df.columns]
        # df = df.reset_index()
        return df   
    
    def make_freq_table(self):
        df = pd.DataFrame(self.statistics_)
        return df    
    
    def plot_pos_freq(self):
        df = pd.concat([pd.DataFrame(self.statistics_[i]['pos_freq_'],index=[i]) for i in self.statistics_])
        df = df.fillna(0)
        df = (df.T/df.sum(1)).reset_index()
        df = pd.melt(df, id_vars=['index'], value_vars=set(df.columns)-set(['index']))
        fig = px.line_polar(df, 
                            r="value",
                            theta="index", 
                            color="variable", 
                            line_close=True,
                            title='Распределение частей речи по коллеуциям текста'
                            )
        return fig

    def plot_pos_types_freq(self, type):
        df = pd.concat([pd.DataFrame(self.statistics_[i][f'{type}'],index=[i]) for i in self.statistics_])
        df = df.fillna(0)
        df = (df.T/df.sum(1)).reset_index()
        df = pd.melt(df, id_vars=['index'], value_vars=set(df.columns)-set(['index']))
        fig = px.line_polar(df, 
                            r="value",
                            theta="index", 
                            color="variable", 
                            line_close=True,
                            title=f'Распределение {type} по коллекциям текста'
                            )
        return fig
    
    def _get_oov(self, doc):
        matcher = Matcher(doc.vocab)
        matcher.add("oov", [[{"POS": "X"}]])
        return matcher(doc, as_spans=True)
    
    def _is_omonim(self, word):
        return len(self.morph.parse(word))>1
    
    def _get_matches(self, doc):
        matcher = Matcher(doc.vocab)
        matcher.add("oov", [[{"POS": "X"}]])
        matcher.add("words", [[{"IS_PUNCT": False}]])
        return matcher(doc,as_spans=False)

    def preview(self,i,k=500):
        return self.corpus[i][:k]

    def calc_morph(self):
        pass
    def calc_lex(self):
        pass