# How to setup the Pedestrian Attribute Recognition System:


#### 1. Download the source code from github:
https://github.com/guofeng201507/capstone_demo_web.git
#### 2. Install all the libraries from requirements.txt

#### 3. Copy the pedestrian detection model weights file (.pth) into the directory ./trained_weights/PETA/img_model, make sure the model name matches the name used in the program pedestrian_attri_recog_model_dpn.py

#### 4. Start the flask application by executing application.py

#### 5. Trigger /init_big_table to get the Sqlite Database created.

#### 6. Execute /mass_load api to mass load all the pedestrian images inside the specific folder (Please specify the path in the code) into the central database, wait for it to finish. (100K images took 5 hours as per our testing)

#### 7. System is loaded with initial datasets and ready for service.
