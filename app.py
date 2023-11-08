import pickle
from gensim.models import KeyedVectors
from flask import Flask, request, render_template
from utils import tokenizador, combinacion_vectores_por_suma, sentimient_TextBlob,sentimient_sia
from googletrans import Translator
import nltk

nltk.download('vader_lexicon')
app = Flask(__name__, template_folder="templates")

@app.before_request
def load_models():
    
    global clasificador
    global w2v_modelo

    w2v_dir = "models/modelo_sg_twitter_300.txt"
    clasificador_dir = "models/lr_sg_twitter.pkl"
    w2v_modelo = KeyedVectors.load_word2vec_format(w2v_dir)
 
    with open(clasificador_dir, "rb") as f:
        clasificador = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    texto_es = next(request.form.values())
    translator = Translator()
    texto_en = translator.translate(texto_es, src="es", dest="en").text

    titulo_tokens = tokenizador(texto_en)
    titulo_vector = combinacion_vectores_por_suma(titulo_tokens, w2v_modelo)
    titulo_categoria = clasificador.predict(titulo_vector)
    output = titulo_categoria[0].capitalize()
    
    output_textBlob_polaridad,output_textBlob_sentimiento = sentimient_TextBlob(texto_en)
    output_sia_polaridad,output_sia_sentimiento = sentimient_sia(texto_en)

    return render_template('index.html',
                            title='"{}"'.format(texto_es), 
                            category=output,
                            # category='Polaridad: {}'.format(output),
                            category_textBlob_polaridad=output_textBlob_polaridad,
                            category_textBlob_sentimiento=output_textBlob_sentimiento,
                            category_sia_polaridad=output_sia_polaridad,
                            category_sia_sentimiento = output_sia_sentimiento
                            )

if __name__ == "__main__":
    app.run(debug=True)