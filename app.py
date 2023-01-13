from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)
model = load_model('swahili.h5')


def preprocess_text():
  pass


@app.route('/')
def index():
  return render_template('templates/index.html')


# @app.route('/classify', methods=['POST'])
# def classify():
#   text = request.form['text']
#   # preprocess the text
#   text = preprocess_text(text)
#   result = model.predict(text)
#   return render_template('result.html', result=result)

if __name__ == '__main__':
  app.run(debug=True)
