from flask import Flask, render_template
import pickle
import json

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image1')
def image1():
    
    return render_template('image1.html', img1 = 'static/result/img1.png', img_labeled1 = 'static/result/img_labeled1.png')

@app.route('/image2')
def image2():
    
    return render_template('image2.html', img2 = 'static/result/img2.png', img_labeled2 = 'static/result/img_labeled2.png')

@app.route('/image3')
def image3():
    
    return render_template('image3.html', img3 = 'static/result/img3.png', img_labeled3 = 'static/result/img_labeled3.png')




@app.route('/count')
def count():
    with open('./static/result/prediction_for_web1.pickle', 'rb') as f:
        prediction_for_web1 = json.loads(pickle.load(f))
    with open('./static/result/prediction_for_web2.pickle', 'rb') as f:        
        prediction_for_web2 = json.loads(pickle.load(f))
    with open('./static/result/prediction_for_web3.pickle', 'rb') as f:        
        prediction_for_web3 = json.loads(pickle.load(f))
    return render_template('count.html', prediction_for_web1 = prediction_for_web1, prediction_for_web2 = prediction_for_web2, prediction_for_web3 = prediction_for_web3)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)

