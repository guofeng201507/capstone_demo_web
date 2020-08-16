import os
import sqlite3
from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import json

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = './static/uploads/'


# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from pedestrian_attri_recog_model import AttrRecogModel

model = AttrRecogModel()


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


def query_db(query, args=()):
    cur = get_db().cursor()
    cur.execute(query, args)

    rv = cur.fetchall()  # Retrive all rows
    cur.close()
    return rv
    # return (rv[0] if rv else None) if one else rv


def update_db(query, args=(), one=False):
    conn = get_db()
    conn.execute(query, args)
    conn.commit()
    # conn.close()
    return "DB updated"


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    result = model.predict_image(filename)
    output_filename = filename[:-4] + '_predicted.png'
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + output_filename), code=301)


@app.route('/init', methods=['POST', 'GET'])
def init_db():
    sql_create_table = """ CREATE TABLE IF NOT EXISTS TB_CASE (
                                          IMAGE_ID text NOT NULL,
                                          ATTRIBUTES json NOT NULL,
                                          TIME_STAMP text NOT NULL
                                      ); """

    update_db(sql_create_table)
    return "DB created!"


@app.route('/mass_load', methods=['POST', 'GET'])
def mass_load_images():
    image_data_path = "./data/images/"
    image_files = [f for f in os.listdir(image_data_path) if f[-4:] == '.jpg' or f[-4:] == '.png']

    for f in image_files:
        print("Loading image " + f)
        image_file = image_data_path + f
        result = model.predict_image_general(image_file)
        detected_attributes = {k: [v] for k, v in result.items() if v > 0.5}
        info = {}
        info[f] = detected_attributes
        attr_json = json.dumps(info)

        image_id = f
        date_stamp = datetime.now()

        tmp_list = [image_id, attr_json, date_stamp]
        update_db("REPLACE INTO TB_CASE (IMAGE_ID, attributes, time_stamp) VALUES (?,?,?);", tmp_list)

    return 'Done!'


@app.route("/search_by_attribute/<attributes>", methods=['GET'])
def search_by_attribute(attributes):
    query_str = "%".join(attributes)
    query_str = "%" + query_str + "%"

    sql_query = """
    SELECT IMAGE_ID, ATTRIBUTES
        FROM TB_CASE, json_each(ATTRIBUTES)
        WHERE
            json_each.value LIKE '%personalLess30%personalMale%upperBodyBlack%lowerBodyBlue%';
    """

    conn = get_db()
    # conn.enable_load_extension(True)
    result = conn.execute(sql_query)
    result


if __name__ == "__main__":
    DATABASE = "./db/demo.db"

    app.run()
