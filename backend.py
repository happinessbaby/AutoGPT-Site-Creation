from flask import Flask, redirect, request, render_template, send_file, send_from_directory, abort, current_app
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from generate_cover_letter import generate_basic_cover_letter
from basic_utils import get_file_name, check_file_type, convert_to_txt
import os
from pathlib import Path


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
resume_path = os.getenv('RESUME_PATH')
cover_letter_path = os.getenv('COVER_LETTER_PATH')
# accept requests that are up to 1MB in size
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.docx', '.txt', '.pdf', '.odt']
# app.config['UPLOAD_PATH'] = 'uploads'
# app.config['RESULT_PATH'] = "results"
# enable CSRF protection
# app.config['DROPZONE_ENABLE_CSRF'] = True



class MyForm(FlaskForm):
    company = StringField('company', validators=[DataRequired()])
    job = StringField('job', validators=[DataRequired()])
    file = FileField('file', validators=[FileRequired(), FileAllowed(['docx', 'txt', 'pdf', 'odt'], 'File type not supported!')])



# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/submit', methods=('GET', 'POST'))
@app.route('/', methods=('GET', 'POST'))
def index():
    # if request.method == 'POST':
    form = MyForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        # return redirect('/success')
        # text1 = request.form['text1']
        # text2 = request.form['text2']
        # text3 = request.form['text3']
        # uploaded_file = request.files['file']
        # # checks if user submits without uploading file
        # if uploaded_file.filename != '':
        # #     uploaded_file.save(uploaded_file.filename)
        # file_ext = os.path.splitext(filename)[1]
        # if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
        #     abort(400)
        file.save(os.path.join(resume_path, filename))
        convert_to_txt(os.path.join(resume_path, filename))
        generate_basic_cover_letter(form.company.data, form.job.data, os.path.join(resume_path, Path(filename).stem+'.txt'))
        return send_file(os.path.join(cover_letter_path, 'cover_letter.txt'), as_attachment=True)
        return redirect('/')
    return render_template('index.html', form = form)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)
    

# @app.route('/results')
# def results():
#     return send_from_directory('static', 'results.html')

if __name__ == '__main__':
    app.run(debug=True)