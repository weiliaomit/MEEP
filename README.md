# MEEP
An MIMIC and eICU extraction pipeline

## 1. Prerequisites 
1).  Both database are hosted on Google Cloud and in order to have access, you can follow the instructions in the following links:  [MIMIV IV](https://physionet.org/content/mimiciv/1.0/) and [eICU](https://eicu-crd.mit.edu/about/eicu/)

2). Set up [Google Cloud](https://cloud.google.com/run/docs/setup) and have a billing project id ready. You can test if you can query MIMIC-IV and eICU database from Google Cloud by running the following test scripts:

    from google.colab import auth
    from google.cloud import bigquery
    import os
    auth.authenticate_user()

    project_id= ###YOUR OWN PROJECT ID###
    os.environ["GOOGLE_CLOUD_PROJECT"]=project_id
    client = bigquery.Client(project=project_id)

    def gcp2df(sql, job_config=None):
	    query = client.query(sql, job_config)
	    results = query.result()
	    return results.to_dataframe() 	
	    
    query = \ 	
    """ 		
        SELECT * 		
        FROM physionet-data.mimic_icu.icustays 		
        LIMIT 1000
    """
    patient = gcp2df(query)

3). File structure:
	 **Resources**: folder containing reource files used in the SQL queries
	 **Output**: folder containing extracted tables
     **utils_mimic_eicu.py**: funtions used to organize queried results 
     **main.py**: main function to run
     **extract_database.py**: extraction SQL scripts
     **Training**: folder containing files in order to train the baseline tasks using the extracted data as well as to perform various model validation
	
## 2. MIMIC-IV and eICU Extraction
Once the data access and Google Cloud is set up, you can start extracting the data. 
1). Under the default setting, you can run the following command and specify your  Google Cloud id:

    python main.py --database MIMIC --project_id xxx
 And 
 

    python main.py --database eICU --project_id xxx
The default eICU extraction will use mean and std from MIMIC to perform z score, to change this:

    python main.py --databse eICU --project_id xxx --norm_eicu eICU

 2). If you want to apply a different age and ICU length of stay filtering, simply run 
 

    python main.py --database MIMIC --project_id xxx --age_min 40 --los_min 12 --los_max 72

 3). If you want to skip the default outlier removal 

    python main.py --database MIMIC --project_id xxx --no_removal

 4). If you want to end the pipeline only after raw record being extracted without further cleaning
 

    python main.py --databse MIMIC --project_id xxx --exit_point Raw
    
 5). If you want to use a specific cohort, such as patients with congestive heart failure
 

    python main.py --database MIMIC --project_id xxx --patient_group CHF
6). If none of the patient groups satisfy your requirement , you can save a custom id file use and use

    python main.py --database MIMIC --project_id xxx --custom_id --customid_dir ./my_group.csv

## 3. Training
(to be added)
## 4. Cross validation 
(to be added)
