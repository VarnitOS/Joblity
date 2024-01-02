import numpy as np
import pandas as pd

from utils.pwd_utils import get_job_vector

def run(args):
    # Generate X - numpy array of shape (n_examples, n_questions), filled with 0s and 1s 
    n_examples = args.n_examples
    n_questions = 8
    X = np.random.choice([0,1], size=(n_examples, n_questions))

    # Create numpy array of shape (n_questions, n_jobs), filled with -1s and 0s
    job_vectors = []
    for qn_num in range(n_questions):
        job_vectors.append(get_job_vector(qn_num))
    job_vectors = np.asarray(job_vectors)

    # Create Y - numpy array of shape (n_examples, n_jobs), filled with 0s (ineligible) and 1s (eligible)
    Y = np.matmul(X, job_vectors)
    Y[Y == 0] = 1                       # For jobs not cancelled, set to 1 (eligible)
    Y[Y < 0] = 0                        # For jobs cancelled, set to 0 (ineligible)

    X_df = pd.DataFrame(X, columns=['Viz','Hear','Wheelchair','Autism','Hands','Strangers','Computers','Creativity'])
    Y_df = pd.DataFrame(Y, columns=['job' + str(num) for num in range(30)])

    X_df.to_csv(args.X_file, index=False)
    Y_df.to_csv(args.Y_file, index=False)


'''
    print(X.shape)
    print(job_vectors.shape)
    print(Y.shape)
'''

