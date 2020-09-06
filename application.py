import json
import os
import sqlite3
from datetime import datetime
import time
from tqdm import tqdm

import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = './static/uploads/'

SCORE_THRESHOLD = 0.5


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
def home():
    return render_template('index.html')


@app.route('/upload')
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
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

        result = model.predict_image(filename)

        detected_attributes = list({k: v for k, v in result.items() if v > SCORE_THRESHOLD}.keys())

        return render_template('upload.html', filename=filename, attributes=detected_attributes)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # conn = get_db()
    # df2.to_sql('TB_BIG_TABLE_DEMO', conn, if_exists='replace', index=False)

    output_filename = filename[:-4] + '_predicted.png'
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + output_filename), code=301)


@app.route('/mass_load', methods=['POST', 'GET'])
def mass_load_images():
    start = time.time()
    image_data_path = "./static/PA100K/"

    image_files = [f for f in os.listdir(image_data_path) if f[-4:] == '.jpg' or f[-4:] == '.png']

    print(f'Mass loading started for {len(image_files)} images')

    df = pd.DataFrame()

    conn = get_db()

    i = 0
    for f in tqdm(image_files):
        image_file = image_data_path + f
        result = model.predict_image_general(image_file)
        detected_attributes = {k: v for k, v in result.items() if v > SCORE_THRESHOLD}

        df2 = pd.DataFrame({k: [v] for k, v in result.items()})
        df2['image_id'] = f
        df2['TIME_STAMP'] = datetime.now()
        df2['attributes'] = json.dumps(detected_attributes)
        if df.empty:
            df = df2
        else:
            df = df.append(df2, ignore_index=True)
        i = i + 1
        if i % 2000 == 0:
            df.to_sql('TB_BIG_TABLE_DEMO', conn, if_exists='append', index=False)
            df = pd.DataFrame(columns=df.columns)
            print(f'2000 rows saved to DB')

    end = time.time()

    print(f'Mass loading completed, took {end - start} seconds')

    return 'Done!'


@app.route('/init_big_table', methods=['POST', 'GET'])
def init_big_table():
    sql_create_table = """ CREATE TABLE IF NOT EXISTS TB_BIG_TABLE_DEMO (
                                          IMAGE_ID text NOT NULL,
                                          personalLess30 int default 0, personalLess45 int default 0, 
                                          personalLess60 int default 0, personalLarger60 int default 0, 
                                          carryingBackpack int default 0, carryingOther int default 0, 
                                          lowerBodyCasual int default 0, upperBodyCasual int default 0,
                                          lowerBodyFormal int default 0, upperBodyFormal int default 0,
                                          
    accessoryHat int default 0, upperBodyJacket int default 0, lowerBodyJeans int default 0,
    footwearLeatherShoes int default 0, upperBodyLogo int default 0, hairLong int default 0, 
    personalMale int default 0, carryingMessengerBag int default 0, accessoryMuffler int default 0,
    accessoryNothing int default 0, carryingNothing int default 0, upperBodyPlaid int default 0,
    carryingPlasticBags int default 0, footwearSandals int default 0, footwearShoes int default 0,
    lowerBodyShorts int default 0, upperBodyShortSleeve int default 0, lowerBodyShortSkirt int default 0,
    footwearSneaker int default 0, upperBodyThinStripes int default 0, accessorySunglasses int default 0,
    lowerBodyTrousers int default 0, upperBodyTshirt int default 0, upperBodyOther  int default 0,
 
  upperBodyVNeck int default 0, upperBodyBlack int default 0, upperBodyBlue int default 0,
    upperBodyBrown int default 0, upperBodyGreen int default 0, upperBodyGrey int default 0,
    upperBodyOrange int default 0, upperBodyPink int default 0, upperBodyPurple int default 0,
    upperBodyRed int default 0, upperBodyWhite int default 0, upperBodyYellow int default 0,
    lowerBodyBlack int default 0, lowerBodyBlue int default 0, lowerBodyBrown int default 0,
    lowerBodyGreen int default 0, lowerBodyGrey int default 0, lowerBodyOrange int default 0,
    lowerBodyPink int default 0, lowerBodyPurple int default 0, lowerBodyRed int default 0,
    lowerBodyWhite int default 0, lowerBodyYellow int default 0, hairBlack int default 0,
    hairBlue int default 0, hairBrown int default 0, hairGreen int default 0,
    hairGrey int default 0, hairOrange int default 0, hairPink int default 0,
    hairPurple int default 0, hairRed int default 0, hairWhite int default 0,
    
    hairYellow int default 0, footwearBlack int default 0, footwearBlue int default 0,
    footwearBrown int default 0, footwearGreen int default 0, footwearGrey int default 0,
    footwearOrange int default 0, footwearPink int default 0, footwearPurple int default 0,
    footwearRed int default 0, footwearWhite int default 0, footwearYellow int default 0,
    accessoryHeadphone int default 0, personalLess15 int default 0, carryingBabyBuggy int default 0,
    hairBald int default 0, footwearBoots int default 0, lowerBodyCapri int default 0,

    carryingShoppingTro int default 0, carryingUmbrella int default 0, personalFemale int default 0,
    carryingFolder int default 0, accessoryHairBand int default 0, lowerBodyHotPants int default 0,
    accessoryKerchief int default 0, lowerBodyLongSkirt int default 0, upperBodyLongSleeve int default 0,
    lowerBodyPlaid int default 0, lowerBodyThinStripes int default 0, carryingLuggageCase int default 0,
    upperBodyNoSleeve int default 0, hairShort int default 0, footwearStocking int default 0,
    upperBodySuit int default 0, carryingSuitcase int default 0, lowerBodySuits int default 0,
    upperBodySweater int default 0, upperBodyThickStripes int default 0, carryingBlack int default 0,
    carryingBlue int default 0, carryingBrown int default 0, carryingGreen int default 0,
    carryingGrey int default 0, carryingOrange int default 0, carryingPink int default 0,
    carryingPurple int default 0, carryingRed int default 0, carryingWhite int default 0,
    carryingYellow int default 0,  attributes text,
                                          TIME_STAMP text NOT NULL
                                      ); """

    update_db(sql_create_table)
    return "DB created!"


@app.route('/search/')
def search_by_attributes():
    return render_template('search_screen.html')


@app.route('/search/', methods=['POST'])
def search():
    selected_fields = request.form.getlist("accessory") + request.form.getlist("carrying") \
                      + request.form.getlist("footwear") + request.form.getlist("hair") \
                      + request.form.getlist("lowerBody") + request.form.getlist("personal") \
                      + request.form.getlist("upperBody")

    result = query_db_based_on_attributes(selected_fields)

    return render_template('image_search_result.html', images_info=result)


@app.route('/search_by_image/', methods=['POST'])
def search_by_image():
    selected_fields_string = request.form.get("attributes")
    selected_fields = json.loads(selected_fields_string.replace('\'', '\"'))

    result = query_db_based_on_attributes(selected_fields)

    return render_template('image_search_result.html', images_info=result)


def query_db_based_on_attributes(selected_fields):
    new_list = []
    for field in selected_fields[:5]:
        new_list.append(field + " > " + str(SCORE_THRESHOLD))
    condition_string = " and ".join(new_list)
    query = "SELECT IMAGE_ID, attributes from TB_BIG_TABLE_DEMO WHERE " + condition_string + " LIMIT 20"
    rows = query_db(query)
    result = []
    for row in rows:
        tmp_dict = {}
        tmp_dict['image_id'] = row[0]
        tmp_dict['attr_list'] = list(json.loads(row[1]).keys())
        result.append(tmp_dict)
    return result


@app.route('/display_returned_images/<filename>')
def display_searched_image(filename):
    return redirect(url_for('static', filename='PA100K/' + filename), code=301)


if __name__ == "__main__":
    DATABASE = "./db/demo_new.db"

    app.run()
