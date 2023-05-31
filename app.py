from flask import Flask, render_template, request, redirect, url_for, send_file
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_cover_letter', methods=['POST'])
def generate_cover_letter():
    my_name = request.form['my_name']
    company_name = request.form['company_name']
    resume_file = request.files['resume_file']
    resume_file.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename))
    os.system(f'python generate_cover_letter.py {my_name} {company_name} {resume_file.filename}')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.run(debug=True) 