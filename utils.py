import numpy as np
import spacy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("es_core_news_sm", disable=['parser', 'tagger', 'ner', 'textcat'])

def tokenizador(texto):
  doc = nlp(texto)
  token_valido = []
  for token in doc:
    valido = not token.is_stop and token.is_alpha
    if valido:
      token_valido.append(token.text.lower())  
  return token_valido

def combinacion_vectores_por_suma(tokens, modelo):
  vector_resultante = np.zeros((1,300))
  for token in tokens:
    try:
      vector_resultante += modelo.get_vector(token)
    except KeyError:
      continue
  return vector_resultante

def sentimient_TextBlob(frase):
    tb = TextBlob(frase)
    polaridad = round(tb.sentiment.polarity,3)
    if polaridad >= 0.05:
        sentiment= "Positivo"
    elif polaridad <= -0.05:
          sentiment= "Negativo"
    else:        sentiment="Neutral"
    return polaridad,sentiment

def sentimient_sia(frase):
   sia = SentimentIntensityAnalyzer()
   sentimiento = sia.polarity_scores(frase)

   polaridad=round(sentimiento['compound'],3)
   if polaridad >= 0.05:
      sentiment= "Positivo"
   elif polaridad <= -0.05:
      sentiment= "Negativo"
   else:
      sentiment="Neutral"

   return polaridad,sentiment
