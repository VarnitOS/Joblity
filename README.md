## Jobility

Jobility is a platform that bridges the employment gap for Persons with Disability (PWDs). It uses AI technology to tailor the matching process between skillful PWDs and suitable employers, and also taps on enabling technologies such as speech recognition for increased convenience for PWDs.

# Features

## 1) Deep Learning Recommender

- Deep Learning job recommendations tailored to PWD disabilities, needs
- AI-generated explanations to build PWD trust in job recommendations
- Details for each job, with suitability for disabilities, and a link to apply

## 2) Speech Recognition

-Real-time speech-to-text translation
-Update profile page for employers to view additional PWD information

## 3) Employee Leaderboard 

-Companies ranked by PWD-friendliness
-Leaderboard rankings can be accessed by PWDs, incentivizing good PR
-Future: automatic ranking by PWD reviews & ratings

# Installation instructions

## Frontend

Note: Before running the following instructions, ensure that npm and node are installed
Note: The speech recognition feature is only supported on Google Chrome, Microsoft Edge and Safari. Please use either of these browsers
Note: To run frontend, copy and paste each line one by one

Instructions are same for Linux / Mac / Windows

cd frontend				# change into frontend/ directory

node --version				# ensure you have at least node ​​v14.15.1	

npm --version      			# ensure your npm version is at least 6.14.4

npm install				# install the required dependencies

npm start				# you should see the application on localhost:3000 

Directory structure:

frontend/

└── public/

└──src/

    └── assets/
    
    └── components/

    └── context/

    └── examples/

    └── layouts/

    └── App.js

    └── index.js

    └── routes.js

Other files not listed, e.g., configuration files



# Backend

Note: Before running the following instructions, ensure that python3 and pip are installed
Note: To run backend, copy and paste each line one by one, depending on your system (Linux, Mac, Windows)

Linux
cd backend					# change into backend directory

python3 -m pip --version			# Ensure python 3 and pip are installed

python3 -m pip install --user virtualenv	# Installing virtualenv

python3 -m venv env				# Create a virtual environment

source env/bin/activate			# Activate the virtual environment

which pip					# Confirm pip is from virtual environment

pip install -r requirements_linux.txt		# Install required packages

flask run					# Run the flask app

Mac

cd backend					# change into backend directory

python3 -m pip --version			# Ensure python 3 and pip are installed

python3 -m pip install --user virtualenv	# Installing virtualenv

python3 -m venv env				# Create a virtual environment

source env/bin/activate			# Activate the virtual environment

which pip					# Confirm pip is from virtual environment

pip install -r requirements_mac.txt		# Install required packages

flask run					# Run the flask app

Windows - Due to windows file and path restrictions, please ensure path to backend/ directory is not too long, and also only contains alphanumeric characters, spaces, hyphens and underscores

cd backend				   	# change into backend directory

python -m pip --version		   	# ensure you have python 3 and pip installed

python -m pip install --user virtualenv		# Installing virtualenv

python -m venv env			   	# Create a virtual environment

.\env\Scripts\activate			   	# Activate the virtual environment

where pip					# Confirm pip is from virtual environment

pip install -r requirements_windows.txt	# Install required packages

flask run				   	# Run the flask app

Directory structure:

backend/

└── data/

└── deploy/

└── step/

└── utils/

└── weights/

└── .gitignore

└── app.py

└── main.py

└── requirements_linux.txt

└── requirements_mac.txt

└── requirements_windows.txt

└── run.sh
