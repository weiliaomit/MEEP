# Set up Google big query
from google.colab import auth
from google.cloud import bigquery
auth.authenticate_user()
# from google.colab import drive
# drive.mount('/content/drive')
import os
from utils_mimic import *

def extract_mimic(args):

    os.environ["GOOGLE_CLOUD_PROJECT"]=args.project_id
    client = bigquery.Client(project=args.project_id)

    def gcp2df(sql, job_config=None):
        query = client.query(sql, job_config)
        results = query.result()
        return results.to_dataframe()

    level_to_change = 1
    ID_COLS = ['subject_id', 'hadm_id', 'stay_id']
    ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']

    to_hours = lambda x: max(0, x.days * 24 + x.seconds // 3600)

    #define our patient cohort by age, icu stay time
    query_d_items = \
    """
    SELECT DISTINCT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.admission_age,
        i.icu_intime,
        i.icu_outtime,
    FROM physionet-data.mimic_derived.icustay_detail i
    WHERE i.hadm_id is not null and i.stay_id is not null
        and i.hospstay_seq = 1
        and i.icustay_seq = 1
        and i.admission_age >= {min_age}
        and (i.icu_outtime >= (i.icu_intime + INTERVAL {min_los} Hour))
        and (i.icu_outtime <= (i.icu_intime + INTERVAL {max_los} Hour))
    ORDER BY subject_id
    ;
    """.format(min_age=args.age_min, min_los=args.los_min, max_los=args.los_max)

    patient= gcp2df(query_d_items)
    # TODO add special group filtering
    icuids_to_keep = patient['stay_id']
    icuids_to_keep = set([str(s) for s in icuids_to_keep])
    subject_to_keep = patient['subject_id']
    subject_to_keep = set([str(s) for s in subject_to_keep])
    # creat template fill_df
    patient.set_index('stay_id', inplace=True)
    patient['max_hours'] = (patient['icu_outtime'] - patient['icu_intime']).apply(to_hours)
    missing_hours_fill = range_unnest(patient, 'max_hours', out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN
    fill_df = patient.reset_index()[ID_COLS].join(missing_hours_fill.set_index('stay_id'), on='stay_id')
    fill_df.set_index(ID_COLS + ['hours_in'], inplace=True)

