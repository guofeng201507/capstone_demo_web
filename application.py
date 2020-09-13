import json
import os
import sqlite3
from datetime import datetime
import time
from tqdm import tqdm

import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
from imutils.object_detection import non_max_suppression

from flask_bootstrap import Bootstrap

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'jpg'])
UPLOAD_FOLDER = './static/uploads/'
video_split_nth_frame = 20
video_split_resize_width = 192
video_split_resize_height = 256

SCORE_THRESHOLD = 0.5

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024

bootstrap = Bootstrap(app)

from pedestrian_attri_recog_model import AttrRecogModel

model = AttrRecogModel()


# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

def sample_video_crop_pedestrian_images(video_name, video_folder, output_folder,
                                        hog, nth_frame, resize_width, resize_height, show=False):
    video_file = os.path.join(video_folder, video_name)
    video_base_name = video_name.split(".")[0]

    image_id = 1
    count = 0
    cap = cv2.VideoCapture(video_file)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        count += 1

        if ret == True:
            if count % nth_frame == 0:
                # width, height = (720, 480)
                # frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)

                (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
                # apply non-maxima suppression to the bounding boxes using a
                # fairly large overlap threshold to try to maintain overlapping
                # boxes that are still people
                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, probs=None, overlapThresh=0.10)

                for (xA, yA, xB, yB) in pick:
                    frame_crop = frame[yA:yB, xA:xB]
                    image_id_str = str(image_id).zfill(5)
                    full_image_name = "_".join([video_base_name, image_id_str]) + ".png"
                    image_id += 1

                    # resize_width, resize_height = 192, 256
                    frame_crop = cv2.resize(frame_crop, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
                    # Write out the cropped image
                    cv2.imwrite(os.path.join(output_folder, full_image_name), frame_crop)

                if show:
                    frame_plot = frame.copy()
                    for (xA, yA, xB, yB) in pick:
                        cv2.rectangle(frame_plot, (xA, yA), (xB, yB), (0, 255, 0), 2)

                    # show the output images
                    cv2.imshow("Detected Pedestrians", frame_plot)

                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
        else:
            break
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


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


@app.route('/walala')
def walala():
    return render_template('test_base_bs.html')


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

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')

        result = model.predict_image(filename)

        detected_attributes = list({k: v for k, v in result.items() if v > SCORE_THRESHOLD}.keys())

        return render_template('upload.html', filename=filename, attributes=detected_attributes)
    else:
        flash('Allowed image types are -> %s' % (', '.join(list(ALLOWED_EXTENSIONS))))
        return redirect(request.url)


@app.route('/upload_video')
def upload_video_form():
    return render_template('upload_video.html')


@app.route('/upload_video', methods=['POST', 'GET'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Video successfully uploaded and displayed')

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # nth_frame = 10
        # resize_width = 192
        # resize_height = 256

        video_name = filename
        video_folder = UPLOAD_FOLDER
        output_folder = os.path.join(UPLOAD_FOLDER, filename.split(".")[0])

        # Create the folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        sample_video_crop_pedestrian_images(video_name, video_folder, output_folder, hog, video_split_nth_frame,
                                            video_split_resize_width, video_split_resize_height, False)

        load_images_from_directory(output_folder)

        return render_template('upload_video.html')
    else:
        flash('Allowed video types are -> %s' % (', '.join(list(ALLOWED_VIDEO_EXTENSIONS))))
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
    return load_images_from_directory()


def load_images_from_directory(image_data_path="./static/PA100K/"):
    start = time.time()
    # image_data_path = "./static/PA100K/"
    if image_data_path[-1] != '/':
        image_data_path = image_data_path + '/'
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
    df.to_sql('TB_BIG_TABLE_DEMO', conn, if_exists='append', index=False)
    print(f'All rows saved to DB')
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

    # return render_template('image_search_result.html', images_info=result)
    return render_template('test_base_bs.html', images_info=result)



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
