import json
from flask import Flask, request, jsonify
from deploy.utils import get_job_listings, get_job_info, get_explanations
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, this is cs206 main'

@app.route('/explanations', methods = ['GET', 'POST'])
def app_get_explanation_for_recommendation():
    # disability_qn_vector = [0,1,1,1,1,1,1,0]
    data = request.get_json()
    print(data)
    disability_qn_vector = data['data']['input']
    if (1 not in disability_qn_vector):
        return "You are suitable for all available jobs! For more tailored recommendations, please answer our questions on the profile page."
    explanations = get_explanations(disability_qn_vector)
    return ' '.join(explanations)

@app.route('/jobs', methods = ['GET', 'POST'])
def app_get_jobs():
    data = request.get_json()
    print(data)
    disability_qn_vector = data['data']['input']

    job_listings = get_job_listings(disability_qn_vector)
    return jsonify(job_listings)

@app.route('/jobs/info', methods = ['GET', 'POST'])
def app_get_job_info():
    data = request.get_json()
    # job_title = request.args.get('job_title')
    # company = request.args.get('company')

    job_title = data['data']['job_title']
    company = data['data']['company']

    print(job_title)
    print(company)
    
    # job_title = 'Mailman'
    # company = 'SingPost'

    return get_job_info(job_title, company)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, threaded=True, debug=True)