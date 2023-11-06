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
    polaridad = tb.sentiment.polarity
    return f'\nLa Polaridad de la frase con TextBlob es: {polaridad:.2f}'

def sentimient_sia(frase):
   sia = SentimentIntensityAnalyzer()
   sentimiento = sia.polarity_scores(frase)

  #  print("\nPuntuación de polaridad con SIA:", sentimiento['compound'])

   if sentimiento['compound'] >= 0.05:
      sentiment= "Sentimiento: Positivo"
   elif sentimiento['compound'] <= -0.05:
      sentiment= "Sentimiento: Negativo"
   else:
      sentiment="Sentimiento: Neutral"

  #  return sentimiento['compound']
   return f'La Polaridad de la frase con sia es: ',sentimiento['compound'],sentiment 