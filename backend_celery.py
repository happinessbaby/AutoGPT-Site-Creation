from flask import Flask, redirect, request, render_template, send_file, send_from_directory, abort, current_app, flash, url_for, config
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from generate_cover_letter import generate_basic_cover_letter
from basic_utils import convert_to_txt
import os
from pathlib import Path
from celery import Celery, Task
from flask_dropzone import Dropzone
from flask_wtf.csrf import CSRFProtect,  CSRFError


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

flask_app = Flask(__name__)
flask_app.secret_key = os.getenv('FLASK_SECRET_KEY')
redis_password=os.getenv('REDIS_PASSWORD')
# redis_password = ''
flask_app.config.from_mapping(
    CELERY=dict(
        broker_url=f'redis://:{redis_password}@localhost:6379/0',
        result_backend=f'redis://:{redis_password}@localhost:6379/0',
        task_ignore_result=True,
    ),
)
def celery_init_app(flask_app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(flask_app.name, task_cls=FlaskTask)
    celery_app.config_from_object(flask_app.config["CELERY"])
    celery_app.set_default()
    flask_app.extensions["celery"] = celery_app
    return celery_app

celery_app = celery_init_app(flask_app)
# CELERY_RESULT_SERIALIZER = 'json'
# CELERY_TASK_SERIALIZER = 'json'

dropzone = Dropzone(flask_app)
csrf = CSRFProtect(flask_app)
csrf.init_app(flask_app)
flask_app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
flask_app.config['DROPZONE_ALLOWED_FILE_TYPE'] = '.docx, .odt, .pdf, .txt'

# app.config['UPLOAD_EXTENSIONS'] = ['.docx', '.txt', '.pdf', '.odt']
flask_app.config['UPLOAD_PATH'] = os.getenv('RESUME_PATH')
flask_app.config['RESULT_PATH'] = os.getenv('COVER_LETTER_PATH')
flask_app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
# enable CSRF protection
flask_app.config['DROPZONE_ENABLE_CSRF'] = True



class MyForm(FlaskForm):
    company = StringField('Company', validators=[DataRequired()])
    job = StringField('Position', validators=[DataRequired()])
    # file = FileField('Resume file', validators=[FileRequired(), FileAllowed(['docx', 'txt', 'pdf', 'odt'], 'File type not supported!')])
    # submit = SubmitField("Generate Cover Letter")



@flask_app.route('/', methods=('GET', 'POST'))
def index(): 
    form = MyForm()
    if request.method == 'POST':
        # Validates form fields
        if form.validate_on_submit():
            filename = ""
            # syntax for dropzone's upload multiple set to true,
            #  see advanced usage: https://readthedocs.org/projects/flask-dropzone/downloads/pdf/latest/
            for key, f in request.files.items():
                if key.startswith('file'):
                    f.save(os.path.join(flask_app.config['UPLOAD_PATH'], f.filename))
                    filename=f.filename
            # file = request.files['file']
            # filename = file.filename
            if filename != '':
                form_data = {
                    'company': form.company.data,
                    'job': form.job.data,
                    'filename': filename,
                    }
                send_cover_letter.delay(form_data)
                return redirect(url_for('index'))
            else:
                abort(400)
    return render_template('index.html', form = form)

@celery_app.task(serializer='json')
def send_cover_letter(form_data):
    # TBA: error handling and logging
    read_path =  os.path.join(flask_app.config['UPLOAD_PATH'], Path(form_data['filename']).stem+'.txt')
    convert_to_txt(os.path.join(flask_app.config['UPLOAD_PATH'], form_data['filename']), read_path)
    if (Path(read_path).exists()):
        generate_basic_cover_letter(form_data['company'], form_data['job'], read_path)
        with flask_app.test_request_context():
            with flask_app.app_context():
                send_file(os.path.join(flask_app.config['RESULT_PATH'], 'cover_letter.txt'))
                # return {"status": True} 
            

@flask_app.route('/static/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    read_path = os.path.join(flask_app.root_path, flask_app.config['RESULT_PATH'])
    return send_from_directory(read_path,
                               filename, as_attachment=True)

@flask_app.route('/loading/', methods=['GET', 'POST'])
def loading_model():
    return render_template ("loading.html")

# handle CSRF error
@flask_app.errorhandler(CSRFError)
def csrf_error(e):
    return e.description, 400




if __name__ == '__main__':

    # # config file has STATIC_FOLDER='/core/static'
    # flask_app.static_url_path=flask_app.config.get('STATIC_FOLDER')

    # # set the absolute path to the static folder
    # flask_app.static_folder=flask_app.root_path + flask_app.static_url_path

    # print(flask_app.static_url_path)
    # print(flask_app.static_folder)
    flask_app.run(debug=True)