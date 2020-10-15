#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, json, request

from joblib import load

model = load("text_classification.joblib")

def requestResults(text):
    tweet = model.predict([text])
    if tweet == 0:
        return "Not-Sarcastic"
    else:
        return "Sarcastic"

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('Index.html')
    
@app.route('/predictSarcasm', methods=['POST'])
def predictSarcasm():
    text= request.form['userText']
    if text == "":
        return render_template('Index.html', prediction_text = '{}'.format(text))
    prediction = requestResults(text)
    return render_template('Index.html', prediction_text = '{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=False)


# In[ ]:




