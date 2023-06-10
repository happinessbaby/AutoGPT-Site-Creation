from flask import Flask, request, render_template, send_file, send_from_directory
from generate_cover_letter import generate_basic_cover_letter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # text1 = request.form['text1']
    text2 = request.form['text2']
    text3 = request.form['text3']
    file = request.files['file']
    # download_link = 'https://example.com/download'
    # with open('cover_letter.txt', 'w') as f:
    #     f.write(f'Dear {name},\n\nThank you for your interest in our company. We have received your application and would like to invite you to an interview.\n\nPlease find attached a copy of your cover letter for your reference.\n\nBest regards,\n[Your Name]')
    #     f.write('\n' + download_link)
    file_name = generate_basic_cover_letter(text2, text3, file)
    return send_file(f'cover_letter_{file_name}.txt', as_attachment=True)

@app.route('/results')
def results():
    return send_from_directory('static', 'results.html')

if __name__ == '__main__':
    app.run(debug=True)