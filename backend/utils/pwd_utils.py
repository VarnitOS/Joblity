def get_list_of_questions():
    questions = [
        'Are you visually impaired?',
        'Do you have hearing problems?',
        'Do you require wheelchairs to move around?',
        'Are you autistic?',
        'Do you face difficulties working with your hands?',
        'Do you face difficulties talking to strangers?',
        'Do you face difficulties working with computers?',
        'Do you face difficulties using your creativity to make something new?'
    ]
    return questions

viz_wheel = ['Flyer Distributer', 'Mailman', 'Delivery Staff', 'Delivery Personnel','Truck Driver']
hear_autism_strangers = ['Cashier', 'Waiter', 'Call Center Worker', 'Customer Service', 'Bank Teller', 'Frontline Sales']
hands = ['Factory/General Worker', 'Food Processing Worker', 'Packing Worker', 'Manufacturing Worker']
computer = ['Data Entry', 'Data Processing Clerk', 'Data Analyst', 'Business Analyst', 'Data Engineer', 'IT Help Desk', 'Web Developer']
creativity = ['Advertising Copywriter', 'Marketing Copywriter', 'Advertising Staff', 'Marketing Staff']
for_everyone = ['Admin Assistant', 'HR Staff', 'Research Assistant', 'Personal Assistant']

def get_full_list_of_jobs():
    return viz_wheel + hear_autism_strangers + hands + computer + creativity + for_everyone

def get_job_vector(question_num):
    list_of_jobs = get_full_list_of_jobs()
    jobs_to_exclude_list = [viz_wheel, hear_autism_strangers, viz_wheel, hear_autism_strangers,
                            hands, hear_autism_strangers, computer, creativity]
    jobs_to_exclude = jobs_to_exclude_list[question_num]

    job_vector = [0] * len(list_of_jobs)
    for job in jobs_to_exclude:
        job_index = list_of_jobs.index(job)
        job_vector[job_index] = -1

    return job_vector

def print_difficulties(x):
    questions = get_list_of_questions()
    difficulties = []
    for i in range(len(x)):
        if x[i] == 1:
            difficulties.append(questions[i])
    print('Difficulties:', difficulties)

def print_negated_jobs(y):
    list_of_jobs = get_full_list_of_jobs()
    negated_jobs = []
    for i in range(len(y)):
        if y[i] == 0:
            negated_jobs.append(list_of_jobs[i])
    print('Negated jobs:', negated_jobs)
    
def get_valid_jobs(y):
    list_of_jobs = get_full_list_of_jobs()
    ret = []
    for i in range(len(y)):
        if y[i] == 1:
            ret.append(list_of_jobs[i])
    return ret

job_titles_to_companies = {
    'Flyer Distributer': ['Flyer Distribution SG', 'NTUC FairPrice', 'AHH.SG', 'Stryker', 'Robert Walters', 'Fidelity International'],
    'Mailman': ['SingPost'],
    'Delivery Staff': ['Qxpress Delivery', 'Ryde', 'SingPost', 'TADA', 'GetVan', 'Grab', 'Lalamove', 'NinjaVan','Pickupp',  'Gojek','GogoX', 'ComfortDelGro', 'UParcel'], 
    'Delivery Personnel':['Lalamove', 'SingPost', 'TADA', 'ComfortDelGro', 'Gojek','GogoX', 'Grab',  'UParcel', 'GetVan', 'NinjaVan','Pickupp', 'Qxpress Delivery', 'Ryde',], 
    'Truck Driver': ['GogoX','SingPost','UParcel', 'GetVan', 'Lalamove', 'NinjaVan','Pickupp', 'Qxpress Delivery',],
    'Cashier': ['After Me Patisserie','NTUC FairPrice', 'POPULAR', 'Hotel G', 'KSC Consultants', 'Green Olive Group', 'Adam Management Services', 'Toptaste Food Management', 'Passion Restaurants', 'Auxiliary Consultant', 'Sai Lee Traders', 'IKEA', 'Cold Storage Singapore', 'An He Trading'],
    'Waiter': ['Tian Tian Holding Pte Ltd', 'Long Xi Investment Pte Ltd', 'Parkway Pantai', 'Taiga Dining', 'Ice-Cold Beer Pte Ltd', 'Grand Hyatt Singapore', 'Galaxy Concepts Pte Ltd'],
    'Call Center Worker': ['Gen Capital', 'Terra Systems Pte Ltd', 'Nityo.com', 'MSD', 'Micron', 'Emerson'],
    'Customer Service': ['Transcendent Business Services', 'Singapore Airlines', 'NTUC Health', 'Adidas', 'BSGroup', 'TransitLink'],
    'Bank Teller': ['Shinhan Bank', 'WaterStone Consulting Pte Ltd', 'Persolkelly Singapore Pte Ltd', 'HSBC', 'Standard Chartered', 'OCBC Bank'],
    'Frontline Sales': ['Adidas', 'Nike', 'Indeed', 'DNA Medical Supplies Pte Ltd', 'The Alchemists Design', 'Siemens', 'Gigworks Pte Ltd', 'Balenciaga Singapore Pte Ltd', 'Hermes Singapore Pte Ltd'],
    'Factory/General Worker': ['FRIENDS', 'Haxton Engineering Pte Ltd', 'Green Wealth Pharmacy Pte Ltd', 'Shalom Transport Services Pte Ltd', 'Kang Logistics Pte Ltd', 'EURO-LINK Consultants'],
    'Food Processing Worker': ['Legaxy Employment', 'Kee Song Food Corporation', 'Song Fish Dealer', 'NTUC FoodFare', 'Raffles Employment Pte Ltd', 'CMY Food Industries Pte Ltd', 'Spring Ocean Food Industries Pte Ltd'], 
    'Packing Worker': ['Green Wealth Pharmacy Pte Ltd', 'Shalom Transport Services Pte Ltd', 'HK Marine Pte Ltd', 'Legaxy Employment', 'RFT Marketing', 'Raffles Employment', 'Oil States International'],
    'Manufacturing Worker': ['Gemini Personnel Pte Ltd', 'Legaxy Employment', 'Rolls-Royce', 'MSD', 'Seawaves Frozen Food Pte Ltd', 'JoinUs Asia Singapore', 'Shell'],
    'Data Entry': ['Yoong Tong Engineering Pte Ltd', 'Planco Pte Ltd', 'Singyi F&B Pte Ltd', 'Elliott Moss Consulting Pte Ltd', 'Mirandah Asia', 'Ultimate Desk Pte Ltd', 'Kian Hock Piling Pte Ltd', 'Toppan Forms Pte Ltd'],
    'Data Processing Clerk': ['Yoong Tong Engineering Pte Ltd', 'Planco Pte Ltd', 'Singyi F&B Pte Ltd', 'Elliott Moss Consulting Pte Ltd', 'Mirandah Asia', 'Ultimate Desk Pte Ltd', 'Kian Hock Piling Pte Ltd', 'Toppan Forms Pte Ltd'],
    'Data Analyst': ['TikTok', 'Singtel', 'Johnson & Johnson', 'Visa', 'InfoCepts', 'Cargill', 'OKBL Pte Ltd', 'Binance', 'EY', 'McKinsey & Company', 'Recruit Express', 'NCS'],
    'Business Analyst': ['Google', 'Hodie Marketing', 'Etihad Airways', 'Pepper Global Pte Ltd', 'Disney Media & Entertainment Distribution', 'Shopee', 'GIC', 'Binance'],
    'Data Engineer': ['Twitter', 'TikTok', 'Opensource Pte Ltd', 'HP', 'JPMorgan Chase Bank', 'Hitachi Systems Network Technologies', 'Zimplistic', 'Dyson', 'Micron', 'Equinix'],
    'IT Help Desk': ['JointHire Singapore Pte Ltd', 'Infinite Computer Solutions Pte Ltd', 'Encora Technologies Pte Ltd', 'Binance', 'Dwellworks Singapore', 'Hitachi Systems Network Technologies Pte Ltd', 'CommonTown Pte Ltd', 'thatz', 'Workiva'], 
    'Web Developer': ['Eco Ring Singapore Pte Ltd', 'JointHire Singapore Pte Ltd', 'Rakuten Asia Pte Ltd', 'Findsgjobs Ltd', 'Alkem Company Pte Ltd', 'Titansoft Singapore', 'Tank Culture'],
    'Advertising Copywriter': ['Tribal Worldwide Pte Ltd', 'Mediatropy Pte Ltd', 'Protocol Pte Ltd', 'Dentsu Solutions Pte Ltd', 'The Teeth Pte Ltd', 'Mullenlowe Singapore Pte Ltd'], 
    'Marketing Copywriter': ['Mullenlowe Singapore Pte Ltd','Tribal Worldwide Pte Ltd', 'Mediatropy Pte Ltd', 'Protocol Pte Ltd', 'Dentsu Solutions Pte Ltd', 'The Teeth Pte Ltd'],
    'Advertising Staff': ['V Medical Aesthetics Pte Ltd', 'Telepathic Private Limited', 'Stone Forest Accountserve Pte Ltd', 'RTB House Pte Ltd', 'Phaidon International', 'Kun Heng Construction', 'Esplanade', 'Collab Asia Sea Pte Ltd'], 
    'Marketing Staff': ['Collab Asia Sea Pte Ltd', 'V Medical Aesthetics Pte Ltd', 'Telepathic Private Limited', 'Stone Forest Accountserve Pte Ltd', 'RTB House Pte Ltd', 'Phaidon International', 'Kun Heng Construction', 'Esplanade'],
    'Admin Assistant': ['Ultimate Desk Pte Ltd', 'Kian Hock Piling Pte Ltd', 'Toppan Forms Pte Ltd', 'Yoong Tong Engineering Pte Ltd', 'Planco Pte Ltd', 'Singyi F&B Pte Ltd', 'Elliott Moss Consulting Pte Ltd', 'Mirandah Asia'],
    'HR Staff': ['Grab', 'Shopee', 'TikTok', 'Apple', 'Recruit Express', 'GIC', 'Goldman Sachs', 'Meta', 'Google', 'Twitter'],
    'Research Assistant': ['Recruit Express', 'Savills IM', 'National University of Singapore', 'Singapore Management University', 'LambdaGen Pte Ltd', 'Phaos Technology', 'Salesforce Singapore', 'Nanyang Technological University'], 
    'Personal Assistant': ['ATO Partners Pte Ltd', 'Robert Half', 'Jobee Pte Ltd', 'Awesome Gaming Pte Ltd', 'Outram', 'Xingang Investment Pte Ltd', 'Fitson Singapore Pte Ltd']
}

def get_valid_companies(job_title):
    return job_titles_to_companies[job_title]
