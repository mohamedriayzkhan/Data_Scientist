from flask import Flask, render_template, request
import joblib
import os
app = Flask(__name__)


model = joblib.load('model.pkl')
interest_map = joblib.load('interest_map.pkl')


interest_map_rev = {}
for interest, number in interest_map.items():
    interest_map_rev[number] = interest

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':     
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        pred_num = model.predict([[age, gender]])[0]

        prediction = interest_map_rev.get(pred_num, "Unknown")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
