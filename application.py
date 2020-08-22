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


@app.route('/init_big_table', methods=['POST', 'GET'])
def init_big_table():
    sql_create_table = """ CREATE TABLE IF NOT EXISTS TB_BIG_TABLE (
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
    carryingYellow int default 0,  
                                          TIME_STAMP text NOT NULL
                                      ); """

    update_db(sql_create_table)
    return "DB created!"


if __name__ == "__main__":
    DATABASE = "./db/demo.db"

    app.run()
