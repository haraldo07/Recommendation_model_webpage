from flask import Flask, request, render_template,session,redirect,url_for
import pickle
import pandas as pd
from model import ItemRecommender

app = Flask(__name__)
import pymysql as pms

conn = pms.connect(host="localhost", port=3306,
                   user="root",
                   password="H4RRYUchiha$",
                   db="Login")
cur = conn.cursor()
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def main():
    return render_template("index2.html")

@app.route('/login', methods=['POST'])
def login():
    
    username = request.form['username']
    password = request.form['password']
    
    cur.execute('SELECT * FROM details WHERE username = %s AND password = %s', (username, password,))
    account = cur.fetchone()
    if account:
        return render_template("success.html")
    else:
        return render_template("index2.html", msg='Incorrect username/password!')
    
@app.route("/predict", methods=['post'])
def pred():
    features = request.form['genre']
    pred = model.predict(features)
    return render_template("Table V01.html",data=pred)    
if __name__=='__main__':
    app.run(port=5000)
    
    
    
    
    
    
    
    
    