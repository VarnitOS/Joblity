import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.pwd_utils import get_valid_jobs, get_valid_companies, print_difficulties
import torch
from utils.net import NeuralNetwork

app = Flask(__name__)
CORS(app)

# Initialize model
model = NeuralNetwork()
# Load weights with CPU device
model.load_state_dict(torch.load('./weights/model_weights.pth', map_location='cpu'))
model.eval()

@app.route('/')
def hello_world():
    return 'Hello, this is cs206 main'

@app.route('/explanations', methods=['GET', 'POST'])
def app_get_explanation_for_recommendation():
    try:
        data = request.get_json()
        print("Received data:", data)
        disability_qn_vector = data['data']['input']
        
        if 1 not in disability_qn_vector:
            return "You are suitable for all available jobs! For more tailored recommendations, please answer our questions on the profile page."
        
        # Convert to tensor and get predictions
        x = torch.tensor(disability_qn_vector).float()
        difficulties = print_difficulties(x)
        
        return jsonify({
            'explanations': difficulties,
            'input_received': disability_qn_vector
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/jobs', methods=['GET', 'POST'])
def app_get_jobs():
    try:
        data = request.get_json()
        print("Received data:", data)
        disability_qn_vector = data['data']['input']
        
        # Convert to tensor and get predictions
        x = torch.tensor(disability_qn_vector).float()
        with torch.no_grad():
            y_pred = model(x)
            y_ints = torch.round(torch.sigmoid(y_pred))
        
        # Get job recommendations
        recommended_jobs = get_valid_jobs(y_ints)
        
        # Get companies for each job
        job_companies = {}
        for job in recommended_jobs:
            job_companies[job] = get_valid_companies(job)
        
        return jsonify({
            'jobs': recommended_jobs,
            'companies': job_companies,
            'input_received': disability_qn_vector
        })
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/jobs/info', methods=['GET', 'POST'])
def app_get_job_info():
    try:
        data = request.get_json()
        job_title = data['data']['job_title']
        company = data['data']['company']
        
        print("Looking up info for:", job_title, "at", company)
        
        # Get job info from jobs_info.json
        with open('./data/jobs_info.json', 'r') as f:
            jobs_data = json.load(f)
            
        if job_title in jobs_data['jobs']:
            return jsonify(jobs_data['jobs'][job_title])
        else:
            return jsonify({'error': 'Job not found'}), 404
            
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)