# How to setup the Pedestrian Attribute Recognition System:


#### 1. Download the source code from github:
https://github.com/guofeng201507/capstone_demo_web.git
#### 2. Install all the libraries from requirements.txt

#### 3. Copy the pedestrian detection model weights file (.pth) into the directory ./trained_weights/PETA/img_model, make sure the model name matches the name used in the program pedestrian_attri_recog_model_dpn.py

#### 4. Start the flask application by executing application.py

#### 5. Copy the demo_new.db file from shared Google Drive into project directory "db"

#### 6. Copy the PA100K images from shared Google Drive into project directory "./static/PA100K"

### Option A: Use the existing pedestrain attributes DB
####     No Action required, system is ready for service

### Option B: Clean the data in the existing database and reload everything
####     B.1. Execute http://localhost:5000/init_big_table to clear all the data in DB
####     B.2. Execute http://localhost:5000/mass_load api to mass load all the pedestrian images inside the specific folder (Please specify the path in the code) into the central database, wait for it to finish. (100K images took 5 hours as per our testing)
####     B.3. System is loaded with initial datasets and ready for service.

### Testing
#### Sample images and video to test the system are placed in directory "DEMO_FILES"
