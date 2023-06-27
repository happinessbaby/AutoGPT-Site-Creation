from flask import Flask, redirect, request, render_template, send_file, send_from_directory, abort, current_app, flash, url_for, config
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from generate_cover_letter import generate_basic_cover_letter
from basic_utils import convert_to_txt
import os
from pathlib import Path
from celery import Celery, Task


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
# app.config['CELERY_BROKER_URL'] = f'redis://:{redis_password}@localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = f'redis://:{redis_password}@localhost:6379/0'
# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)
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

# accept requests that are up to 1MB in size
# flask_app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
# app.config['UPLOAD_EXTENSIONS'] = ['.docx', '.txt', '.pdf', '.odt']
flask_app.config['UPLOAD_PATH'] = os.getenv('RESUME_PATH')
flask_app.config['RESULT_PATH'] = os.getenv('COVER_LETTER_PATH')
flask_app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
# enable CSRF protection
# app.config['DROPZONE_ENABLE_CSRF'] = True


class MyForm(FlaskForm):
    company = StringField('Company', validators=[DataRequired()])
    job = StringField('Position', validators=[DataRequired()])
    file = FileField('Resume file', validators=[FileRequired(), FileAllowed(['docx', 'txt', 'pdf', 'odt'], 'File type not supported!')])
    submit = SubmitField("Generate Cover Letter")



@flask_app.route('/', methods=('GET', 'POST'))
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
        file.save(os.path.join(flask_app.config['UPLOAD_PATH'], filename))
        form_data = {
        'company': form.company.data,
        'job': form.job.data,
        'filename': filename
        }
        send_cover_letter.delay(form_data)
        redirect(url_for('static', filename="cover_letter.txt"))
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
                return {"status": True} 
            

@flask_app.route('/static/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    read_path = os.path.join(flask_app.root_path, flask_app.config['RESULT_PATH'])
    return send_from_directory(read_path,
                               filename, as_attachment=True)

@flask_app.route('/loading/', methods=['GET', 'POST'])
def loading_model():
    return render_template ("loading.html")




if __name__ == '__main__':

    # # config file has STATIC_FOLDER='/core/static'
    # flask_app.static_url_path=flask_app.config.get('STATIC_FOLDER')

    # # set the absolute path to the static folder
    # flask_app.static_folder=flask_app.root_path + flask_app.static_url_path

    # print(flask_app.static_url_path)
    # print(flask_app.static_folder)
    flask_app.run(debug=True)