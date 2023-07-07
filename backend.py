from flask import Flask, redirect, request, render_template, send_file, send_from_directory, abort, current_app, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from generate_cover_letter import generate_basic_cover_letter
from basic_utils import convert_to_txt
import os
from pathlib import Path
from flask_dropzone import Dropzone
from flask_wtf.csrf import CSRFProtect,  CSRFError


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
app.config['UPLOAD_PATH'] = os.getenv('RESUME_PATH')
app.config['RESULT_PATH'] = os.getenv('COVER_LETTER_PATH')
# app.config['UPLOAD_EXTENSIONS'] = ['.docx', '.txt', '.pdf', '.odt']
dropzone = Dropzone(app)
csrf = CSRFProtect(app)
csrf.init_app(app)



class MyForm(FlaskForm):
    company = StringField('company', validators=[DataRequired()])
    job = StringField('job', validators=[DataRequired()])


@app.route('/', methods=('GET', 'POST'))
def index():
    form = MyForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            filename = ""
            for key, f in request.files.items():
                if key.startswith('file'):
                    f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
                    filename=f.filename
            if filename != '':
                form_data = {
                'company': form.company.data,
                'job': form.job.data,
                'filename': filename,
                }
                read_path =  os.path.join(app.config['UPLOAD_PATH'], Path(filename).stem+'.txt')
                convert_to_txt(os.path.join(app.config['UPLOAD_PATH'], form_data['filename']), read_path)
                if (Path(read_path).exists()):
                    generate_basic_cover_letter(form_data['company'], form_data['job'], read_path)
                    with app.test_request_context():
                        with app.app_context():
                            send_file(os.path.join(app.config['RESULT_PATH'], 'cover_letter.txt'))
                            return redirect(url_for('index'))
            else:
                abort(400)
    return render_template('index.html', form = form)

@app.route('/static/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    read_path = os.path.join(app.root_path, app.config['RESULT_PATH'])
    return send_from_directory(read_path,
                               filename, as_attachment=True)

@app.route('/loading/', methods=['GET', 'POST'])
def loading_model():
    return render_template ("loading.html")

# handle CSRF error
@app.errorhandler(CSRFError)
def csrf_error(e):
    return e.description, 400

if __name__ == '__main__':
    app.run(debug=True)