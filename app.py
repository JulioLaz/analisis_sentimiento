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
    output_textBlob = sentimient_TextBlob(texto_en)
    output_sia = sentimient_sia(texto_en)

    return render_template('index.html',
                            title='Frase: {}'.format(texto_es), 
                            category='Categoría: {}'.format(output),
                            category_textBlob='Categoría: {}'.format(output_textBlob),
                            # category_sia=float(output_sia))
                            category_sia='Categoría: {}'.format(output_sia))

if __name__ == "__main__":
    app.run(debug=True)
