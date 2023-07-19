from flask import Flask, redirect, request, render_template, send_file, send_from_directory, abort, current_app, flash, url_for, config, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from werkzeug.security import check_password_hash, generate_password_hash
from generate_cover_letter import generate_basic_cover_letter
from basic_utils import convert_to_txt, check_content_safety
import os
from pathlib import Path
from celery import Celery, Task
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask_dropzone import Dropzone
from flask_wtf.csrf import CSRFProtect,  CSRFError
from flask_session import Session
from flask_login import LoginManager
from flask_login import current_user, login_required, logout_user
from flask_simplelogin import Message, SimpleLogin, login_required
# from backend_streamlit import create_chatbot
import redis
import uuid
import json



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

# app.config['UPLOAD_EXTENSIONS'] = ['.docx', '.txt', '.pdf', '.odt']
flask_app.config['UPLOAD_PATH'] = os.getenv('RESUME_PATH')
flask_app.config['RESULT_PATH'] = os.getenv('COVER_LETTER_PATH')
# enable CSRF protection
flask_app.config['DROPZONE_ENABLE_CSRF'] = True
# logger = get_task_logger(__name__)

# Configure Redis for storing the session data on the server-side
flask_app.config['SESSION_TYPE'] = 'redis'
flask_app.config['SESSION_PERMANENT'] = False
flask_app.config['SESSION_USE_SIGNER'] = True
flask_app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')

# Create and initialize the Flask-Session object AFTER `app` has been configured
server_session=Session(flask_app)

# LoginManager is needed for our application
# to be able to log in and out users
# login_manager = LoginManager()
# login_manager.init_app(flask_app)

SimpleLogin(flask_app)



class MyForm(FlaskForm):
    company = StringField('Company', validators=[DataRequired()])
    job = StringField('Position', validators=[DataRequired()])

# class User():
#       # init method or constructor
#     def __init__(self, id=None):
#         self.id = id



@flask_app.route('/', methods=('GET', 'POST'))
def index(): 
    form = MyForm()
    # user = User(id = str(uuid.uuid4()))
    if request.method == 'POST':
        # Validates form fields
        if form.validate_on_submit():
            # syntax for dropzone's upload multiple set to true,
            #  see advanced usage: https://readthedocs.org/projects/flask-dropzone/downloads/pdf/latest/
            for key, f in request.files.items():
                if key.startswith('file'):
                    # Change filename to userid
                    session['id'] = str(uuid.uuid4())
                    file_ext = Path(f.filename).suffix
                    filename = session['id']+file_ext
                    f.filename = filename
                    f.save(os.path.join(flask_app.config['UPLOAD_PATH'], f.filename))
                    form_data = {
                        'company': form.company.data,
                        'job': form.job.data,
                        'filename': filename,
                        }
                    generate.delay(form_data)
                    return redirect(url_for('loading', filename=filename))
    return render_template('index.html', form = form)


@celery_app.task(serializer='json', bind='True')
def generate(self, form_data):
    # TBA: error handling and logging
    with flask_app.test_request_context():
        with flask_app.app_context():
            # task_id = self.request.id
            read_path =  os.path.join(flask_app.config['UPLOAD_PATH'], Path(form_data['filename']).stem+'.txt')
            convert_to_txt(os.path.join(flask_app.config['UPLOAD_PATH'], form_data['filename']), read_path)
            if (Path(read_path).exists()):
                # Check for content safety
                if (check_content_safety(file=read_path)):
                    save_path = os.path.join(flask_app.config['RESULT_PATH'], os.path.basename(read_path))
                    generate_basic_cover_letter(form_data['company'], form_data['job'], read_path=read_path, save_path=save_path)



# @flask_app.route('/downloads/<path:filename>', methods=['GET', 'POST'])
# def download(filename):
#     filename = session['id']
#     read_path = os.path.join(flask_app.root_path, flask_app.config['RESULT_PATH'])
#     return send_file(os.path.join(read_path, filename+".txt"), download_name="cover_letter.txt", as_attachment=True)   
                    # return send_from_directory(read_path,
                    #                filename, as_attachment=True)


@flask_app.route('/loading/', methods=['GET', 'POST'])
def loading():
    if request.method=='GET':
        filename = session['id']+'.txt'
        return render_template ("loading.html", filename = filename)
        

# handle CSRF error
@flask_app.errorhandler(CSRFError)
def csrf_error(e):
    return e.description, 400


# def create_user(**data):
#     """Creates user with encrypted password"""
#     if "username" not in data or "password" not in data:
#         raise ValueError("username and password are required.")

#     # Hash the user password
#     data["password"] = generate_password_hash(
#         data.pop("password"), method="pbkdf2:sha256"
#     )

#     # Here you insert the `data` in your users database
#     # for this simple example we are recording in a json file
#     db_users = json.load(open("users.json"))
#     # add the new created user to json
#     db_users[data["username"]] = data
#     # commit changes to database
#     json.dump(db_users, open("users.json", "w"))
#     return data


# def validate_login(user):
#     db_users = json.load(open("users.json"))
#     if not db_users.get(user["username"]):
#         return False
#     stored_password = db_users[user["username"]]["password"]
#     if check_password_hash(stored_password, user["password"]):
#         return True
#     return False

# def configure_extensions(app):
#     messages = {
#         "login_success": "Welcome!",
#         "is_logged_in": Message("already logged in", "success"),
#         "logout": None,
#     }
#     SimpleLogin(app, login_checker=validate_login, messages=messages)
#     if not os.path.exists("users.json"):
#         with open("users.json", "a") as json_file:
#             # This just touch create a new dbfile
#             json.dump({"username": "", "password": ""}, json_file)


if __name__ == '__main__':

    # # config file has STATIC_FOLDER='/core/static'
    # flask_app.static_url_path=flask_app.config.get('STATIC_FOLDER')

    # # set the absolute path to the static folder
    # flask_app.static_folder=flask_app.root_path + flask_app.static_url_path

    # print(flask_app.static_url_path)
    # print(flask_app.static_folder)
    flask_app.run(debug=True)