from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
  model = pickle.load(file)


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
