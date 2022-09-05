# Set up Google big query
from google.colab import auth
from google.cloud import bigquery
import os
import numpy as np
import pandas as pd
from utils_mimic import *

auth.authenticate_user()


def extract_mimic(args):
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)

    def gcp2df(sql, job_config=None):
        que = client.query(sql, job_config)
        results = que.result()
        return results.to_dataframe()

    # level_to_change = 1
    ID_COLS = ['subject_id', 'hadm_id', 'stay_id']
    ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']

    # datatime format to hour
    to_hours = lambda x: max(0, x.days * 24 + x.seconds // 3600)

    def get_group_id(args):
        if args.patient_group == 'sepsis-3':
            query = \
                """
                SELECT  stay_id
                FROM physionet-data.mimic_derived.sepsis3
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['stay_id']])
        elif args.patient_group == 'ARF':
            query = \
                """
                SELECT DISTINCT stay_id
                FROM physionet-data.mimic_icu.chartevents
                WHERE itemid = 224700 or  itemid = 220339

                UNION ALL

                SELECT i.stay_id
                FROM physionet-data.mimic_hosp.labevents l
                LEFT JOIN physionet-data.mimic_icu.icustays i on l.subject_id = i.subject_id 
                WHERE l.itemid = 50819 
                AND l.charttime between i.intime and i.outtime 

                UNION ALL 

                SELECT DISTINCT v.stay_id 
                FROM physionet-data.mimic_derived.ventilation v
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['stay_id']])
        elif args.patient_group == 'Shock':
            query = \
                """
                SELECT DISTINCT stay_id
                FROM physionet-data.mimic_derived.vasoactive_agent
                WHERE norepinephrine is not null 
                OR epinephrine is not null 
                OR dopamine is not null 
                OR vasopressin is not null 
                OR phenylephrine  is not null 
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['stay_id']])
        elif args.patient_group == 'CHF':
            query = \
                """
                SELECT DISTINCT i.stay_id
                FROM physionet-data.mimic_derived.charlson c
                LEFT JOIN physionet-data.mimic_icu.icustays i on c.hadm_id = i.hadm_id 
                WHERE c.congestive_heart_failure = 1 and i.stay_id is not null
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['stay_id']])
        elif args.patient_group == 'COPD':
            query = \
                """
                SELECT DISTINCT i.stay_id
                FROM physionet-data.mimic_derived.charlson c
                LEFT JOIN physionet-data.mimic_icu.icustays i on c.hadm_id = i.hadm_id 
                WHERE c.chronic_pulmonary_disease = 1 and i.stay_id is not null
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['stay_id']])
        elif args.custom_id == True:
            custom_ids = pd.read_csv(args.customid_dir)
            group_stay_ids = set([str(s) for s in custom_ids['stay_id']])

        return group_stay_ids

    # define our patient cohort by age, icu stay time
    if args.patient_group != 'Generic':
        query = \
            """
            SELECT DISTINCT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.gender,
                i.admission_age as age,
                i.ethnicity,
                i.hospital_expire_flag,
                i.hospstay_seq,
                i.los_icu,
                i.admittime,
                i.dischtime,
                i.icu_intime,
                i.icu_outtime,
                a.admission_type,
                a.insurance,
                a.deathtime,
                a.discharge_location,
                CASE when a.deathtime between i.icu_intime and i.icu_outtime THEN 1 ELSE 0 END AS mort_icu,
                CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
                COALESCE(f.readmission_30, 0) AS readmission_30
            FROM physionet-data.mimic_derived.icustay_detail i
                INNER JOIN physionet-data.mimic_core.admissions a ON i.hadm_id = a.hadm_id
                INNER JOIN physionet-data.mimic_icu.icustays s ON i.stay_id = s.stay_id
                LEFT OUTER JOIN (SELECT d.stay_id, 1 as readmission_30
                            FROM physionet-data.mimic_icu.icustays c, physionet-data.mimic_icu.icustays d
                            WHERE c.subject_id=d.subject_id
                            AND c.stay_id > d.stay_id
                            AND c.intime - d.outtime <= INTERVAL 30 DAY
                            AND c.outtime = (SELECT MIN(e.outtime) from physionet-data.mimic_icu.icustays e 
                                            WHERE e.subject_id=c.subject_id
                                            AND e.intime>d.outtime) ) f
                            ON i.stay_id=f.stay_id
            WHERE i.hadm_id is not null and i.stay_id is not null and i.stay_id in ({group_icuids})
                and i.hospstay_seq = 1
                and i.icustay_seq = 1
                and i.admission_age >= {min_age}
                and (i.icu_outtime >= (i.icu_intime + INTERVAL {min_los} Hour))
                and (i.icu_outtime <= (i.icu_intime + INTERVAL {max_los} Hour))
            ORDER BY subject_id
            ;
            """.format(group_icuids=','.join(get_group_id(args)), min_age=args.age_min, min_los=args.los_min,
                       max_los=args.los_max)
        patient = gcp2df(query)
    else:
        query = \
            """
            SELECT DISTINCT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.gender,
                i.admission_age as age,
                i.ethnicity,
                i.hospital_expire_flag,
                i.hospstay_seq,
                i.los_icu,
                i.admittime,
                i.dischtime,
                i.icu_intime,
                i.icu_outtime,
                a.admission_type,
                a.insurance,
                a.deathtime,
                a.discharge_location,
                CASE when a.deathtime between i.icu_intime and i.icu_outtime THEN 1 ELSE 0 END AS mort_icu,
                CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
                COALESCE(f.readmission_30, 0) AS readmission_30
            FROM physionet-data.mimic_derived.icustay_detail i
                INNER JOIN physionet-data.mimic_core.admissions a ON i.hadm_id = a.hadm_id
                INNER JOIN physionet-data.mimic_icu.icustays s ON i.stay_id = s.stay_id
                LEFT OUTER JOIN (SELECT d.stay_id, 1 as readmission_30
                            FROM physionet-data.mimic_icu.icustays c, physionet-data.mimic_icu.icustays d
                            WHERE c.subject_id=d.subject_id
                            AND c.stay_id > d.stay_id
                            AND c.intime - d.outtime <= INTERVAL 30 DAY
                            AND c.outtime = (SELECT MIN(e.outtime) from physionet-data.mimic_icu.icustays e 
                                            WHERE e.subject_id=c.subject_id
                                            AND e.intime>d.outtime) ) f
                            ON i.stay_id=f.stay_id
            WHERE i.hadm_id is not null and i.stay_id is not null
                and i.hospstay_seq = 1
                and i.icustay_seq = 1
                and i.admission_age >= {min_age}
                and (i.icu_outtime >= (i.icu_intime + INTERVAL {min_los} Hour))
                and (i.icu_outtime <= (i.icu_intime + INTERVAL {max_los} Hour))
            ORDER BY subject_id
            ;
            """.format(min_age=args.age_min, min_los=args.los_min, max_los=args.los_max)

        patient = gcp2df(query)
        # TODO add special group filtering
    icuids_to_keep = patient['stay_id']
    icuids_to_keep = set([str(s) for s in icuids_to_keep])
    subject_to_keep = patient['subject_id']
    subject_to_keep = set([str(s) for s in subject_to_keep])
    # create template fill_df with time window for each stay based on icu in/out time
    patient.set_index('stay_id', inplace=True)
    patient['max_hours'] = (patient['icu_outtime'] - patient['icu_intime']).apply(to_hours)
    missing_hours_fill = range_unnest(patient, 'max_hours', out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN
    fill_df = patient.reset_index()[ID_COLS].join(missing_hours_fill.set_index('stay_id'), on='stay_id')
    fill_df.set_index(ID_COLS + ['hours_in'], inplace=True)

    # start with mimic_derived_data
    # query bg table
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.bg b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.subject_id = i.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    bg = gcp2df(query)

    # initial process bg table
    bg['hours_in'] = (bg['charttime'] - bg['icu_intime']).apply(to_hours)
    bg.drop(columns=['charttime', 'icu_intime', 'aado2_calc'], inplace=True)
    bg = process_query_results(bg, fill_df)

    # query vital sign
    query = """
    SELECT b.*, i.hadm_id, i.icu_intime
    FROM physionet-data.mimic_derived.vitalsign b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.stay_id = i.stay_id
    where b.stay_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime
    """.format(icuids=','.join(icuids_to_keep))
    vitalsign = gcp2df(query)

    # temperature/glucose is a repeat name but different itemid, rename for now and combine later
    vitalsign.rename(columns={'temperature': 'temp_vital'}, inplace=True)
    vitalsign.rename(columns={'glucose': 'glucose_vital'}, inplace=True)
    vitalsign['hours_in'] = (vitalsign['charttime'] - vitalsign['icu_intime']).apply(to_hours)
    vitalsign.drop(columns=['charttime', 'icu_intime', 'temperature_site'], inplace=True)
    vitalsign = process_query_results(vitalsign, fill_df)

    # query blood differential
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.blood_differential b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    blood_diff = gcp2df(query)

    blood_diff['hours_in'] = (blood_diff['charttime'] - blood_diff['icu_intime']).apply(to_hours)
    blood_diff.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    blood_diff = process_query_results(blood_diff, fill_df)

    # query cardiac marker
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.cardiac_marker b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.subject_id = i.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    cardiac_marker = gcp2df(query)

    cardiac_marker['troponin_t'].replace(to_replace=[None], value=np.nan, inplace=True)
    cardiac_marker['troponin_t'] = pd.to_numeric(cardiac_marker['troponin_t'])
    cardiac_marker['hours_in'] = (cardiac_marker['charttime'] - cardiac_marker['icu_intime']).apply(to_hours)
    cardiac_marker.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    cardiac_marker = process_query_results(cardiac_marker, fill_df)

    # query chemistry
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.chemistry b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime
    """.format(icuids=','.join(subject_to_keep))
    chemistry = gcp2df(query)

    # rename glucose into glucose_chem and others
    chemistry.rename(columns={'glucose': 'glucose_chem'}, inplace=True)
    chemistry.rename(columns={'bicarbonate': 'bicarbonate_chem'}, inplace=True)
    chemistry.rename(columns={'chloride': 'chloride_chem'}, inplace=True)
    chemistry.rename(columns={'calcium': 'calcium_chem'}, inplace=True)
    chemistry.rename(columns={'potassium': 'potassium_chem'}, inplace=True)
    chemistry.rename(columns={'sodium': 'sodium_chem'}, inplace=True)

    chemistry['hours_in'] = (chemistry['charttime'] - chemistry['icu_intime']).apply(to_hours)
    chemistry.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    chemistry = process_query_results(chemistry, fill_df)

    # query coagulation
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.coagulation b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    coagulation = gcp2df(query)

    coagulation['hours_in'] = (coagulation['charttime'] - coagulation['icu_intime']).apply(to_hours)
    coagulation.drop(columns=['charttime', 'icu_intime', 'specimen_id'], inplace=True)
    coagulation = process_query_results(coagulation, fill_df)

    # query cbc
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.complete_blood_count b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON b.subject_id = i.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    cbc = gcp2df(query)

    cbc.rename(columns={'hematocrit': 'hematocrit_cbc'}, inplace=True)
    cbc.rename(columns={'hemoglobin': 'hemoglobin_cbc'}, inplace=True)
    # also drop wbc since it's a repeat 51301
    cbc['hours_in'] = (cbc['charttime'] - cbc['icu_intime']).apply(to_hours)
    cbc.drop(columns=['charttime', 'icu_intime', 'specimen_id', 'wbc'], inplace=True)
    cbc = process_query_results(cbc, fill_df)

    # query culture
    query = """
    SELECT b.subject_id, b.charttime, b.specimen, b.screen, b.positive_culture, b.has_sensitivity, 
    i.hadm_id, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.culture b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    culture = gcp2df(query)

    culture.rename(columns={'specimen': 'specimen_culture'}, inplace=True)
    culture['hours_in'] = (culture['charttime'] - culture['icu_intime']).apply(to_hours)
    culture.drop(columns=['charttime', 'icu_intime'], inplace=True)
    culture = culture.groupby(ID_COLS + ['hours_in']).agg(['last'])
    culture = culture.reindex(fill_df.index)

    # query enzyme
    query = """
    SELECT b.*, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.enzyme b
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = b.subject_id
    where b.subject_id in ({icuids})
    and b.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    enzyme = gcp2df(query)

    # also drop ck_mb since it's a repeat 50911
    enzyme['hours_in'] = (enzyme['charttime'] - enzyme['icu_intime']).apply(to_hours)
    enzyme.drop(columns=['charttime', 'icu_intime', 'specimen_id', 'ck_mb'], inplace=True)
    enzyme = process_query_results(enzyme, fill_df)

    # query gcs
    query = """
    SELECT g.subject_id, g.stay_id, g.charttime, g.gcs, i.hadm_id, i.icu_intime
    FROM physionet-data.mimic_derived.gcs g
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = g.stay_id
    where g.stay_id in ({icuids})
    and g.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(icuids_to_keep))

    gcs = gcp2df(query)

    gcs['hours_in'] = (gcs['charttime'] - gcs['icu_intime']).apply(to_hours)
    gcs.drop(columns=['charttime', 'icu_intime'], inplace=True)
    gcs = process_query_results(gcs, fill_df)

    # query inflammation
    query = """
    SELECT g.subject_id, g.hadm_id, g.charttime, g.crp, i.stay_id, i.icu_intime
    FROM physionet-data.mimic_derived.inflammation g 
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = g.subject_id
    where g.subject_id in ({icuids})
    and g.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(subject_to_keep))
    inflammation = gcp2df(query)

    inflammation['hours_in'] = (inflammation['charttime'] - inflammation['icu_intime']).apply(to_hours)
    inflammation.drop(columns=['charttime', 'icu_intime'], inplace=True)
    inflammation = process_query_results(inflammation, fill_df)

    # query uo
    query = """
    SELECT g.stay_id, g.charttime, g.weight, g.uo, i.icu_intime, i.subject_id, i.hadm_id
    FROM physionet-data.mimic_derived.urine_output_rate g 
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = g.stay_id
    where g.stay_id in ({icuids})
    and g.charttime between i.icu_intime and i.icu_outtime

    """.format(icuids=','.join(icuids_to_keep))
    uo = gcp2df(query)
    uo['hours_in'] = (uo['charttime'] - uo['icu_intime']).apply(to_hours)
    uo.drop(columns=['charttime', 'icu_intime'], inplace=True)
    uo = process_query_results(uo, fill_df)

    # join and save
    # use MIMIC-Extract way to query other itemids that was present in MIMIC-Extract
    # load resources
    chartitems_to_keep = pd.read_excel('./resources/chartitems_to_keep_0505.xlsx')
    lab_to_keep = pd.read_excel('./resources/labitems_to_keep_0505.xlsx')
    var_map = pd.read_csv('./resources/Chart_makeup_0505 - var_map0505.csv')
    chart_items = chartitems_to_keep['chartitems_to_keep'].tolist()
    lab_items = lab_to_keep['labitems_to_keep'].tolist()
    chart_items = set([str(i) for i in chart_items])
    lab_items = set([str(i) for i in lab_items])

    query = \
        """
        SELECT c.subject_id, i.hadm_id, c.stay_id, c.charttime, c.itemid, c.value, c.valueuom
        FROM `physionet-data.mimic_derived.icustay_detail` i
        INNER JOIN `physionet-data.mimic_icu.chartevents` c ON i.stay_id = c.stay_id
        WHERE c.stay_id IN ({icuids})
            AND c.itemid IN ({chitem})
            AND c.charttime between i.icu_intime and i.icu_outtime
            AND c.valuenum is not null
    
        UNION ALL
    
        SELECT DISTINCT i.subject_id, i.hadm_id, i.stay_id, l.charttime, l.itemid, l.value, l.valueuom
        FROM `physionet-data.mimic_derived.icustay_detail` i
        INNER JOIN `physionet-data.mimic_hosp.labevents` l ON i.hadm_id = l.hadm_id
        WHERE i.stay_id  IN ({icuids})
            and l.itemid  IN ({labitem})
            and l.charttime between i.icu_intime and i.icu_outtime
            and l.valuenum > 0
        ;
        """.format(icuids=','.join(icuids_to_keep), chitem=','.join(chart_items), labitem=','.join(lab_items))

    chart_lab = gcp2df(query)

    chart_lab['value'] = pd.to_numeric(chart_lab['value'], 'coerce')
    chart_lab = chart_lab.set_index('stay_id').join(patient[['icu_intime']])
    chart_lab['hours_in'] = (chart_lab['charttime'] - chart_lab['icu_intime']).apply(to_hours)
    chart_lab.drop(columns=['charttime', 'icu_intime'], inplace=True)
    chart_lab.set_index('itemid', append=True, inplace=True)
    var_map.set_index('itemid', inplace=True)
    chart_lab = chart_lab.join(var_map, on='itemid').set_index(['LEVEL1', 'LEVEL2'], append=True)
    chart_lab.index.names = ['stay_id', chart_lab.index.names[1], chart_lab.index.names[2], chart_lab.index.names[3]]
    group_item_cols = ['LEVEL2']
    chart_lab = chart_lab.groupby(ID_COLS + group_item_cols + ['hours_in']).agg(['mean', 'count'])

    chart_lab.columns = chart_lab.columns.droplevel(0)
    chart_lab.columns.names = ['Aggregation Function']
    chart_lab = chart_lab.unstack(level=group_item_cols)
    chart_lab.columns = chart_lab.columns.reorder_levels(order=group_item_cols + ['Aggregation Function'])

    chart_lab = chart_lab.reindex(fill_df.index)
    chart_lab = chart_lab.sort_index(axis=1, level=0)
    new_cols = chart_lab.columns.reindex(['mean', 'count'], level=1)
    chart_lab = chart_lab.reindex(columns=new_cols[0])

    total = bg.join(
        [vitalsign, blood_diff, cardiac_marker, chemistry, coagulation, cbc, culture, enzyme, gcs, inflammation, uo])

    # drop some columns (not well-populated or dependent on existing columns )
    columns_to_drop = ['rdwsd', 'aado2', 'pao2fio2ratio', 'carboxyhemoglobin',
                       'methemoglobin', 'globulin', 'd_dimer', 'thrombin', 'basophils_abs', 'eosinophils_abs',
                       'lymphocytes_abs', 'monocytes_abs', 'neutrophils_abs']
    for c in columns_to_drop:
        total.drop(c, axis=1, level=0, inplace=True)

    idx = pd.IndexSlice
    chart_lab.loc[:, idx[:, ['count']]] = chart_lab.loc[:, idx[:, ['count']]].fillna(0)
    total.loc[:, idx[:, ['count']]] = total.loc[:, idx[:, ['count']]].fillna(0)

    # combine columns since they were from different itemids
    names_to_combine = [
        ['so2', 'spo2'], ['fio2', 'fio2_chartevents'], ['bicarbonate', 'bicarbonate_chem'],
        ['hematocrit', 'hematocrit_cbc'], ['hemoglobin', 'hemoglobin_cbc'], ['chloride', 'chloride_chem'],
        ['glucose', 'glucose_chem'], ['glucose', 'glucose_vital'],
        ['temperature', 'temp_vital'], ['sodium', 'sodium_chem'], ['potassium', 'potassium_chem']
    ]
    for names in names_to_combine:
        original = total.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
        makeups = total.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
        total.drop(names[1], axis=1, level=0, inplace=True)

    csite_map = {'WORM': 'cul_site13', 'CORNEAL EYE SCRAPINGS': 'cul_site13', 'PROSTHETIC JOINT FLUID': 'cul_site13',
                 'Touch Prep/Sections': 'cul_site13', 'Isolate': 'cul_site13', 'Infection Control Yeast': 'cul_site13',
                 'SCOTCH TAPE PREP/PADDLE': 'cul_site13', 'ARTHROPOD': 'cul_site13',
                 'RAPID RESPIRATORY VIRAL ANTIGEN TEST': 'cul_site2',
                 'CHORIONIC VILLUS SAMPLE': 'cul_site13', 'PERIPHERAL BLOOD LYMPHOCYTES': 'cul_site13',
                 'EAR': 'cul_site13',
                 'URINE,KIDNEY': 'cul_site11', 'BRONCHIAL BRUSH': 'cul_site11', 'BIOPSY': 'cul_site13',
                 'NEOPLASTIC BLOOD': 'cul_site13',
                 'AMNIOTIC FLUID': 'cul_site13', 'EYE': 'cul_site13', 'FOOT CULTURE': 'cul_site13',
                 'ASPIRATE': 'cul_site13',
                 'POSTMORTEM CULTURE': 'cul_site13', 'DIALYSIS FLUID': 'cul_site13', 'NAIL SCRAPINGS': 'cul_site13',
                 'DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS': 'cul_site10',
                 'Direct Antigen Test for Herpes Simplex Virus Types 1 & 2': 'cul_site10',
                 'SKIN SCRAPINGS': 'cul_site13',
                 'FOREIGN BODY': 'cul_site13', 'BILE': 'cul_site13', 'BRONCHIAL WASHINGS': 'cul_site11',
                 'Stem Cell - Blood Culture': 'cul_site13',
                 'BONE MARROW': 'cul_site13', 'Influenza A/B by DFA': 'cul_site13', 'JOINT FLUID': 'cul_site13',
                 'ABSCESS': 'cul_site13',
                 'CATHETER OR LINE': 'cul_site13', 'PLEURAL FLUID': 'cul_site13', 'FLUID,OTHER': 'cul_site13',
                 'BRONCHOALVEOLAR LAVAGE': 'cul_site11',
                 'ANORECTAL/VAGINAL CULTURE': 'cul_site12', 'Rapid Respiratory Viral Screen & Culture': 'cul_site2',
                 'THROAT': 'cul_site2',
                 'PERITONEAL FLUID': 'cul_site9', 'Staph aureus Screen': 'cul_site7', 'CSF;SPINAL FLUID': 'cul_site8',
                 'VIRAL CULTURE': 'cul_site10',
                 'TISSUE': 'cul_site6', 'IMMUNOLOGY': 'cul_site5', 'SPUTUM': 'cul_site4', 'STOOL': 'cul_site3',
                 'MRSA SCREEN': 'cul_site7',
                 'SWAB': 'cul_site2', 'SEROLOGY/BLOOD': 'cul_site0', 'URINE': 'cul_site1', 'BLOOD': 'cul_site0',
                 'FLUID WOUND': 'cul_site13',
                 'TRACHEAL ASPIRATE': 'cul_site11', 'POST-MORTEM VIRAL CULTURE': 'cul_site13'}
    total.loc[:, idx['specimen_culture', ['last']]] = pd.Series(
        np.squeeze(total.loc[:, idx['specimen_culture', ['last']]].values)).map(csite_map).values

    # drop Eosinophils
    chart_lab.drop('Eosinophils', axis=1, level=0, inplace=True)
    # combine in chart_lab table
    names = ['Phosphate', 'Phosphorous']
    original = chart_lab.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
    makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    chart_lab.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
    chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    names = ['Potassium', 'Potassium serum']
    original = chart_lab.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
    makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    chart_lab.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
    chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    # Combine between chartlab and total table
    names_list = [
        ['alt', 'Alanine aminotransferase'], ['albumin', 'Albumin'], ['alp', 'Alkaline phosphate'],
        ['aniongap', 'Anion gap'], ['ast', 'Asparate aminotransferase'], ['bicarbonate', 'Bicarbonate'],
        ['bilirubin_total', 'Bilirubin'], ['bilirubin_direct', 'Bilirubin_direct'], ['calcium', 'Calcium ionized'],
        ['chloride', 'Chloride'], ['creatinine', 'Creatinine'],
        ['fibrinogen', 'Fibrinogen'], ['hematocrit', 'Hematocrit'], ['hemoglobin', 'Hemoglobin'],
        ['so2', 'Oxygen saturation'], ['pco2', 'Partial pressure of carbon dioxide'],
        ['platelet', 'Platelets'], ['potassium', 'Potassium'], ['inr', 'Prothrombin time INR'],
        ['pt', 'Prothrombin time PT'],
        ['resp_rate', 'Respiratory rate'], ['sodium', 'Sodium'], ['wbc', 'White blood cell count'],
        ['lactate', 'Lactic acid'], ['ph', 'pH'], ['bun', 'Blood urea nitrogen']]

    for names in names_list:
        original = total.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
        makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
        chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    # In eicu mbp contains both invasive and non-invasive, so combine them
    names_list = [['dbp', 'Diastolic blood pressure'], ['dbp_ni', 'Diastolic blood pressure'],
                  ['mbp', 'Mean blood pressure'], ['mbp_ni', 'Mean blood pressure'],
                  ['sbp', 'Systolic blood pressure'], ['sbp_ni', 'Systolic blood pressure']]

    for names in names_list:
        original = total.loc[:, idx[names[0], ['count', 'mean']]].copy(deep=True)
        makeups = chart_lab.loc[:, idx[names[1], ['count', 'mean']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['count', 'mean']]] = filled.loc[:, ['count', 'mean']].values
        # Xm.drop(names[1], axis=1, level=0, inplace=True)
    chart_lab.drop('Mean blood pressure', axis=1, level=0, inplace=True)
    chart_lab.drop('Diastolic blood pressure', axis=1, level=0, inplace=True)
    chart_lab.drop('Systolic blood pressure', axis=1, level=0, inplace=True)

    columns_to_drop = ['Albumin ascites', 'Albumin pleural', 'Albumin urine', 'Calcium urine', 'Chloride urine',
                       'Cholesterol', 'Cholesterol HDL', 'Cholesterol LDL', 'Creatinine ascites',
                       'Creatinine body fluid',
                       'Creatinine pleural', 'Lactate dehydrogenase pleural', 'Lymphocytes ascites',
                       'Lymphocytes body fluid',
                       'Lymphocytes percent', 'Lymphocytes pleural', 'Red blood cell count ascites',
                       'Red blood cell count pleural']

    for c in columns_to_drop:
        chart_lab.drop(c, axis=1, level=0, inplace=True)

    vital = total.join(chart_lab)

    # screen and positive culture needs impute, they are last columns but with float data type
    vital_encode = pd.get_dummies(vital)

    vital_encode[('positive_culture', 'mask')] = (~vital_encode[('positive_culture', 'last')].isnull()).astype(float)
    vital_encode[('screen', 'mask')] = (~vital_encode[('screen', 'last')].isnull()).astype(float)
    vital_encode[('has_sensitivity', 'mask')] = (~vital_encode[('has_sensitivity', 'last')].isnull()).astype(float)
    # X_encode.fillna(value=0, inplace=True)
    # vital_encode.fillna(value=0, inplace=True)

    col = vital_encode.columns.to_list()
    col.insert(col.index(('screen', 'last')) + 1, ('screen', 'mask'))
    col.insert(col.index(('positive_culture', 'last')) + 1, ('positive_culture', 'mask'))
    col.insert(col.index(('has_sensitivity', 'last')) + 1, ('has_sensitivity', 'mask'))

    vital_final = vital_encode[col[:-3]]

    # start query intervention
    query = """
    select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.endtime, i.icu_intime, i.icu_outtime
    FROM physionet-data.mimic_derived.icustay_detail i
    INNER JOIN physionet-data.mimic_derived.ventilation v ON i.stay_id = v.stay_id
    where v.stay_id in ({icuids})
    and v.starttime < i.icu_outtime
    and v.endtime > i.icu_intime
    """.format(icuids=','.join(icuids_to_keep))

    vent_data = gcp2df(query)
    vent_data = compile_intervention(vent_data, 'vent')

    ids_with = vent_data['stay_id']
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = (ids_all - ids_with)
    novent_data = patient.copy(deep=True)
    novent_data = novent_data.reset_index()
    novent_data = novent_data.set_index('stay_id')
    novent_data = novent_data.iloc[novent_data.index.isin(ids_without)]
    novent_data = novent_data.reset_index()
    novent_data = novent_data[['subject_id', 'hadm_id', 'stay_id', 'max_hours']]
    # novent_data['max_hours'] = novent_data['stay_id'].map(icustay_timediff)
    novent_data = novent_data.groupby('stay_id')
    novent_data = novent_data.apply(add_blank_indicators)
    novent_data.rename(columns={'on': 'vent'}, inplace=True)
    novent_data = novent_data.reset_index()

    # Concatenate all the data vertically
    intervention = pd.concat([vent_data[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'vent']],
                              novent_data[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'vent']]],
                             axis=0)

    # query antibiotics
    query = """
    select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.stoptime as endtime, v.antibiotic, 
    v.route, i.icu_intime, i.icu_outtime 
    FROM physionet-data.mimic_derived.icustay_detail i
    INNER JOIN physionet-data.mimic_derived.antibiotic v ON i.stay_id = v.stay_id
    where v.stay_id in ({icuids})
    and v.starttime < i.icu_outtime 
    and v.stoptime > i.icu_intime 
    ;
    """.format(icuids=','.join(icuids_to_keep))

    antibiotics = gcp2df(query)
    antibiotics = compile_intervention(antibiotics, 'antibiotics')
    intervention = intervention.merge(
        antibiotics[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'antibiotic', 'route']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # vaso agents
    column_names = ['dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine', 'vasopressin', 'dobutamine',
                    'milrinone']

    # TODO(mmd): This section doesn't work. What is its purpose?
    for c in column_names:
        # TOTAL VASOPRESSOR DATA
        query = """
        select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.endtime, i.icu_intime, i.icu_outtime, 
        FROM physionet-data.mimic_derived.icustay_detail i
        INNER JOIN physionet-data.mimic_derived.vasoactive_agent v ON i.stay_id = v.stay_id
        where v.stay_id in ({icuids})
        and v.starttime  < i.icu_outtime
        and v.endtime > i.icu_intime 
        and v.{drug_name} is not null
        ;
        """.format(icuids=','.join(icuids_to_keep), drug_name=c)

        # job_config = bigquery.QueryJobConfig(query_parameters=[
        #     bigquery.ScalarQueryParameter("NAME", "STRING", c)])

        new_data = gcp2df(query)
        new_data = compile_intervention(new_data, c)
        # new_data = continuous_outcome_processing(new_data, patient, icustay_timediff)
        # new_data = new_data.apply(add_outcome_indicators)
        # new_data.rename(columns={'on': c}, inplace=True)
        # new_data = new_data.reset_index()
        # c may not be in Y if we are only extracting a subset of the population, in which c was never
        # performed.

        intervention = intervention.merge(
            new_data[['subject_id', 'hadm_id', 'stay_id', 'hours_in', c]],
            on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
            how='left'
        )

    # heparin
    query = \
        """
    SELECT he.subject_id, he.starttime, he.stoptime as endtime, i.hadm_id, i.stay_id, i.icu_intime, i.icu_outtime
    FROM physionet-data.mimic_derived.heparin he
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.subject_id = he.subject_id
    WHERE he.subject_id in ({ids}) 
    AND  he.starttime < i.icu_outtime
    AND  he.stoptime > i.icu_intime 
    """.format(ids=','.join(subject_to_keep))
    heparin = gcp2df(query)

    heparin = compile_intervention(heparin, 'heparin')
    intervention = intervention.merge(
        heparin[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'heparin']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # crrt
    query = \
        """
    SELECT cr.stay_id, MIN(cr.charttime) as starttime, MAX(cr.charttime) as endtime, i.subject_id, 
    i.hadm_id, i.icu_intime, i.icu_outtime
    FROM physionet-data.mimic_derived.crrt cr
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = cr.stay_id
    WHERE cr.stay_id in ({ids}) 
    AND  cr.charttime BETWEEN i.icu_intime AND i.icu_outtime
    GROUP BY cr.stay_id, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime

    """.format(ids=','.join(icuids_to_keep))
    crrt = gcp2df(query)

    crrt = compile_intervention(crrt, 'crrt')
    intervention = intervention.merge(
        crrt[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'crrt']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # rbc transfusion
    query = \
        """
        WITH rbc as 
            (SELECT amount
            , amountuom
            , stay_id
            , starttime
            , endtime
            FROM physionet-data.mimic_icu.inputevents
            WHERE (itemid in
            (
            225168,  --Packed Red Blood Cells
            226368, --OR Packed RBC Intake
            227070 --PACU Packed RBC Intake
            )
            AND amount > 0
            AND stay_id in ({ids}) 
            )
            ORDER BY stay_id, endtime)
    
        SELECT rbc.stay_id, rbc.starttime, rbc.endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        FROM rbc
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = rbc.stay_id
        WHERE starttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        ORDER BY stay_id
        """.format(ids=','.join(icuids_to_keep))
    rbc_trans = gcp2df(query)

    rbc_trans = compile_intervention(rbc_trans, 'rbc_trans')
    intervention = intervention.merge(
        rbc_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'rbc_trans']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # platelets transfusion
    query = \
        """
        WITH pll as (
            SELECT amount
            , amountuom
            , stay_id
            , starttime
            , endtime
            FROM physionet-data.mimic_icu.inputevents
            WHERE (itemid in
            (
                225170,  --Platelets
                226369, --OR Platelet Intake
                227071  --PACU Platelet Intake
            )
            AND amount > 0
            AND stay_id in ({ids}) 
            )
            ORDER BY stay_id, endtime)
    
        SELECT pll.stay_id, pll.starttime, pll.endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        FROM pll
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = pll.stay_id
        WHERE starttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        ORDER BY stay_id
        """.format(ids=','.join(icuids_to_keep))
    platelets_trans = gcp2df(query)
    platelets_trans = compile_intervention(platelets_trans, 'platelats_trans')
    intervention = intervention.merge(
        platelets_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'platelats_trans']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # ffp transfusion
    query = \
        """
        WITH ffp as (
            SELECT amount
            , amountuom
            , stay_id
            , starttime
            , endtime
            FROM physionet-data.mimic_icu.inputevents
            WHERE (itemid in
            (
                220970,  -- Fresh Frozen Plasma
                226367,  -- OR FFP Intake
                227072  -- PACU FFP Intake
            )
            AND amount > 0
            AND stay_id in ({ids}) 
            )
            ORDER BY stay_id, endtime)
    
        SELECT ffp.stay_id, ffp.starttime, ffp.endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        FROM ffp
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = ffp.stay_id
        WHERE starttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        ORDER BY stay_id
        """.format(ids=','.join(icuids_to_keep))
    ffp_trans = gcp2df(query)
    ffp_trans = compile_intervention(ffp_trans, 'ffp_trans')
    intervention = intervention.merge(
        ffp_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'ffp_trans']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # other infusion
    query = \
        """
        with coll as
            (
            select
                mv.stay_id
            , mv.starttime as charttime
            , mv.endtime as endtime
            -- standardize the units to millilitres
            -- also metavision has floating point precision.. but we only care down to the mL
            , round(case
                when mv.amountuom = 'L'
                    then mv.amount * 1000.0
                when mv.amountuom = 'ml'
                    then mv.amount
                else null end) as amount
            from physionet-data.mimic_icu.inputevents mv
            where mv.itemid in
            (
                220864, --  Albumin 5%  7466 132 7466
                220862, --  Albumin 25% 9851 174 9851
                225174, --  Hetastarch (Hespan) 6%  82 1 82
                225795, --  Dextran 40  38 3 38
                225796  --  Dextran 70
                -- below ITEMIDs not in use
            -- 220861 | Albumin (Human) 20%
            -- 220863 | Albumin (Human) 4%
            )
            and mv.statusdescription != 'Rewritten'
            and
            -- in MetaVision, these ITEMIDs never appear with a null rate
            -- so it is sufficient to check the rate is > 100
                (
                (mv.rateuom = 'mL/hour' and mv.rate > 100)
                OR (mv.rateuom = 'mL/min' and mv.rate > (100/60.0))
                OR (mv.rateuom = 'mL/kg/hour' and (mv.rate*mv.patientweight) > 100)
                )
            and stay_id in ({ids}) 
            )
        -- remove carevue 
        -- some colloids are charted in chartevents
    
        select coll.stay_id, coll.charttime as starttime, coll.endtime, coll.amount as colloid_bolus, 
        i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        from coll
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = coll.stay_id
        -- just because the rate was high enough, does *not* mean the final amount was
        WHERE charttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        --group by coll.stay_id, coll.charttime, coll.endtime
        order by stay_id, charttime 
        """.format(ids=','.join(icuids_to_keep))
    colloid_bolus = gcp2df(query)

    colloid_bolus = compile_intervention(colloid_bolus, 'colloid_bolus')
    intervention = intervention.merge(
        colloid_bolus[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'colloid_bolus']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    # other infusion
    query = \
        """
        with crys as
            (
            select
                mv.stay_id
            , mv.starttime as charttime
            , mv.endtime 
            -- standardize the units to millilitres
            -- also metavision has floating point precision.. but we only care down to the mL
            , round(case
                when mv.amountuom = 'L'
                    then mv.amount * 1000.0
                when mv.amountuom = 'ml'
                    then mv.amount
                else null end) as amount
            from physionet-data.mimic_icu.inputevents mv
            where mv.itemid in
            (
                -- 225943 Solution
                225158, -- NaCl 0.9%
                225828, -- LR
                225944, -- Sterile Water
                225797, -- Free Water
                225159, -- NaCl 0.45%
                -- 225161, -- NaCl 3% (Hypertonic Saline)
                225823, -- D5 1/2NS
                225825, -- D5NS
                225827, -- D5LR
                225941, -- D5 1/4NS
                226089 -- Piggyback
            )
            and mv.statusdescription != 'Rewritten'
            and
            -- in MetaVision, these ITEMIDs appear with a null rate IFF endtime=starttime + 1 minute
            -- so it is sufficient to:
            --    (1) check the rate is > 240 if it exists or
            --    (2) ensure the rate is null and amount > 240 ml
                (
                (mv.rate is not null and mv.rateuom = 'mL/hour' and mv.rate > 248)
                OR (mv.rate is not null and mv.rateuom = 'mL/min' and mv.rate > (248/60.0))
                OR (mv.rate is null and mv.amountuom = 'L' and mv.amount > 0.248)
                OR (mv.rate is null and mv.amountuom = 'ml' and mv.amount > 248)
                )
            )
    
        select crys.stay_id, crys.charttime as starttime, crys.endtime, crys.amount as crystalloid_bolus, 
        i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
        from crys
        INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.stay_id = crys.stay_id
        WHERE charttime < i.icu_outtime
        AND  endtime > i.icu_intime 
        --group by coll.stay_id, coll.charttime, coll.endtime
        order by stay_id, charttime;
    
        """.format(ids=','.join(icuids_to_keep))
    crystalloid_bolus = gcp2df(query)

    crystalloid_bolus = compile_intervention(crystalloid_bolus, 'crystalloid_bolus')
    intervention = intervention.merge(
        crystalloid_bolus[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'crystalloid_bolus']],
        on=['subject_id', 'hadm_id', 'stay_id', 'hours_in'],
        how='left'
    )

    intervention.drop('route', axis=1, inplace=True)
    # for each column, astype to int and fill na with 0
    intervention = intervention.fillna(0)
    intervention.loc[:, 'antibiotic'] = intervention.loc[:, 'antibiotic'].mask(intervention.loc[:, 'antibiotic'] != 0,
                                                                               1).values
    for i in range(5, 20):
        intervention.iloc[:, i] = intervention.iloc[:, i].astype(int)

    intervention.set_index(ID_COLS + ['hours_in'], inplace=True)
    intervention.sort_index(level=['stay_id', 'hours_in'], inplace=True)

    # static info
    #  query patients anchor year
    query = """
    select i.subject_id, i.hadm_id, i.stay_id, i.icu_intime, i.icu_outtime, v.anchor_year, v.anchor_year_group
    FROM physionet-data.mimic_derived.icustay_detail i
    INNER JOIN physionet-data.mimic_core.patients v ON i.subject_id = v.subject_id
    where i.stay_id in ({icuids})
    ;
    """.format(icuids=','.join(icuids_to_keep))
    anchor_year = gcp2df(query)

    query = """
    select c.subject_id, c.hadm_id, i.stay_id, c.myocardial_infarct, c.congestive_heart_failure, 
    c.peripheral_vascular_disease, c.cerebrovascular_disease, c.dementia, c.chronic_pulmonary_disease, 
    c.rheumatic_disease, c.peptic_ulcer_disease, c.mild_liver_disease, c.diabetes_without_cc, 
    c.diabetes_with_cc, c.paraplegia, c.renal_disease, c.malignant_cancer, c.severe_liver_disease, 
    c.metastatic_solid_tumor, c.aids
    FROM physionet-data.mimic_derived.charlson c
    INNER JOIN physionet-data.mimic_derived.icustay_detail i ON i.hadm_id = c.hadm_id
    where i.stay_id in ({icuids})
    """.format(icuids=','.join(icuids_to_keep))
    comorbidity = gcp2df(query)

    patient.reset_index(inplace=True)
    patient.set_index(ID_COLS, inplace=True)
    comorbidity.set_index(ID_COLS, inplace=True)
    anchor_year.set_index(ID_COLS, inplace=True)
    static = patient.join([comorbidity, anchor_year['anchor_year_group']])

    vital_final.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_vital.h5'), key='mimic_vital')
    static.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_static.h5'), key='mimic_static')
    intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_MIMIC_inv.h5'), key='mimic_inv')

    # remove outliers
    total_cols = vital_final.columns.tolist()
    mean_col = [i for i in total_cols if 'mean' in i]
    X_mean = vital_final.loc[:, mean_col]

    range_dict_high = {'so2': 100, 'po2': 770, 'pco2': 220, 'ph': 10, 'baseexcess': 100, 'bicarbonate': 66,
                       'chloride': 200, 'hemoglobin': 30, 'hematocrit': 100, 'calcium': 1.87, 'temperature': 47,
                       'potassium': 15, 'sodium': 250, 'lactate': 33, 'glucose': 2200, 'heart_rate': 390, 'sbp': 375,
                       'sbp_ni': 375, 'mbp': 375, 'mbp_ni': 375, 'dbp': 375, 'dbp_ni': 375, 'resp_rate': 330,
                       'wbc': 1100,
                       'atypical_lymphocytes': 17, 'bun': 300, 'calcium_chem': 28, 'fibrinogen': 1709, 'Phosphate': 22,
                       'Positive end-expiratory pressure': 30, 'ck_cpk': 10000, 'ggt': 10000,
                       'Peak inspiratory pressure': 40,
                       'Magnesium': 22, 'Plateau Pressure': 61, 'Tidal Volume Observed': 2000, 'nrbc': 143, 'inr': 15,
                       'pt': 150, 'mch': 46, 'mchc': 43, 'troponin_t': 24, 'albumin': 60, 'aniongap': 55,
                       'creatinine': 66,
                       'platelet': 2200, 'alt': 11000, 'ast': 22000, 'alp': 4000, 'ld_ldh': 35000,
                       'bilirubin_total': 66,
                       'bilirubin_indirect': 66, 'bilirubin_direct': 66, 'weight': 550, 'uo': 2445,
                       'Central Venous Pressure': 400}
    range_dict_low = {'ph': 6.3, 'baseexcess': -100, 'temperature': 14.2, 'lactate': 0.01, 'uo': 0, 'pH urine': 3}

    for var_to_remove in range_dict_high:
        remove_outliers_h(vital_final, X_mean, var_to_remove, range_dict_high[var_to_remove])
    for var_to_remove in range_dict_low:
        remove_outliers_l(vital_final, X_mean, var_to_remove, range_dict_low[var_to_remove])

    # normalize
    count_col = [i for i in total_cols if 'count' in i]
    col_means, col_stds = vital_final.loc[:, mean_col].mean(axis=0), vital_final.loc[:, mean_col].std(axis=0)
    vital_final.loc[:, mean_col] = (vital_final.loc[:, mean_col] - col_means) / col_stds
    icustay_means = vital_final.loc[:, mean_col].groupby(ID_COLS).mean()
    # impute
    vital_final.loc[:, mean_col] = vital_final.loc[:, mean_col].groupby(ID_COLS).fillna(method='ffill').groupby(
        ID_COLS).fillna(
        icustay_means).fillna(0)
    # 0 or 1
    vital_final.loc[:, count_col] = (vital_final.loc[:, count_col] > 0).astype(float)
    # at this satge only 3 last columns has nan values
    vital_final = vital_final.fillna(0)

    # split data
    stays_v = set(vital_final.index.get_level_values(2).values)
    stays_static = set(static.index.get_level_values(2).values)
    stays_int = set(intervention.index.get_level_values(2).values)
    assert stays_v == stays_static, "Subject ID pools differ!"
    assert stays_v == stays_int, "Subject ID pools differ!"
    train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
    SEED = 41
    np.random.seed(SEED)
    subjects, N = np.random.permutation(list(stays_v)), len(stays_v)
    N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
    train_stay = list(stays_v)[:N_train]
    dev_stay = list(stays_v)[N_train:N_train + N_dev]
    test_stay = list(stays_v)[N_train + N_dev:]

    [(vital_train, vital_dev, vital_test), (Y_train, Y_dev, Y_test), (static_train, static_dev, static_test)] = [
        [df[df.index.get_level_values(2).isin(s)] for s in (train_stay, dev_stay, test_stay)] \
        for df in (vital_final, intervention, static)]

    vital_train.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='vital_train')
    vital_dev.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='vital_dev')
    vital_test.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='vital_test')
    Y_train.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='inv_train')
    Y_dev.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='inv_dev')
    Y_test.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='inv_test')
    static_train.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='static_train')
    static_dev.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='static_dev')
    static_test.to_hdf(os.path.join(args.output_dir, 'MIMIC_split.hdf5'), key='static_test')
    return


def extract_eicu(args):
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)

    def gcp2df(sql, job_config=None):
        que = client.query(sql, job_config)
        results = que.result()
        return results.to_dataframe()

    # level_to_change = 1
    ID_COLS = ['patientunitstayid']
    # minutes to hour
    to_hours = lambda x: int(x // 60)

    def get_group_id(args):
        if args.patient_group == 'sepsis-3':
            sepsis3_ids = pd.read_csv('./resources/sepsis3_eicu.csv')
            group_stay_ids = set([str(s) for s in sepsis3_ids['patientunitstayid']])
        elif args.patient_group == 'ARF':
            query = \
                """
                SELECT DISTINCT l.patientunitstayid
                FROM physionet-data.eicu_crd.lab l
                WHERE l.labname = 'PEEP' 
                and l.labresult >= 0 
                and l.labresult <= 30
                
                UNION ALL
                
                SELECT DISTINCT vt.patientunitstayid 
                FROM (
                    SELECT Distinct i.patientunitstayid, i.priorventstartoffset, i.priorventendoffset	
                    From physionet-data.eicu_crd.respiratorycare i
                    WHERE i.priorventstartoffset >0 or i.priorventendoffset	>0
                    Order by patientunitstayid
                ) vt
                INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = vt.patientunitstayid
                WHERE  vt.priorventstartoffset is not null 
                AND vt.priorventendoffset is not null
                AND FLOOR(LEAST(vt.priorventendoffset, i.unitdischargeoffset)/60) > FLOOR(GREATEST(vt.priorventstartoffset, 0)/60)
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
        elif args.patient_group == 'Shock':
            query = \
                """
                SELECT DISTINCT pm.patientunitstayid, 
                FROM physionet-data.eicu_crd_derived.pivoted_med pm
                INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = pm.patientunitstayid
                WHERE (pm.dopamine = 1 or pm.norepinephrine = 1 or pm.epinephrine = 1 or pm.vasopressin = 1 or pm. phenylephrine = 1)
                AND pm.drugorderoffset is not null 
                AND pm.drugstopoffset is not null
                AND FLOOR(LEAST(pm.drugstopoffset, i.unitdischargeoffset)/60) > FLOOR(GREATEST(pm.drugorderoffset, 0)/60)
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
        elif args.patient_group == 'CHF':
            query = \
                """
                SELECT DISTINCT ad.patientunitstayid,
                FROM physionet-data.eicu_crd.diagnosis ad
                WHERE SUBSTR(ad.icd9code, 1, 3) = '428'
                OR SUBSTR(ad.icd9code, 1, 6) IN ('398.91','402.01','402.11','402.91','404.01','404.03',
                                        '404.11','404.13','404.91','404.93')
                OR SUBSTR(ad.icd9code, 1, 5) BETWEEN '425.4' AND '425.9'
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
        elif args.patient_group == 'COPD':
            query = \
                """
                SELECT DISTINCT ad.patientunitstayid,
                FROM physionet-data.eicu_crd.diagnosis ad
                WHERE SUBSTR(ad.icd9code, 1, 3) BETWEEN '490' AND '505'
                OR SUBSTR(ad.icd9code, 1, 5) IN ('416.8','416.9','506.4','508.1','508.8')
                """
            id_df = gcp2df(query)
            group_stay_ids = set([str(s) for s in id_df['patientunitstayid']])
        elif args.custom_id:
            custom_ids = pd.read_csv(args.customid_dir)
            group_stay_ids = set([str(s) for s in custom_ids['stay_id']])

        return group_stay_ids

    if args.patient_group != 'Generic':
        query = \
            """
            SELECT i.patientunitstayid, i.gender, i.age, i.ethnicity,  
                    CASE WHEN lower(i.hospitaldischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.hospitaldischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS hosp_mort,
                    ROUND(i.unitdischargeoffset/60) AS icu_los_hours, i.hospitaladmitoffset, i.hospitaldischargeoffset,
                   i.unitdischargeoffset, i.hospitaladmitsource, i.unitdischargelocation, 
                   CASE WHEN lower(i.unitdischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.unitdischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS icu_mort, i.hospitaldischargeyear, i.hospitalid      
            From physionet-data.eicu_crd.patient i
            WHERE ROUND(i.unitdischargeoffset/60) Between {min_los} and {max_los} 
            AND patientunitstayid in ({group_icuids})
            AND age not in ({young_age})
            """.format(group_icuids=','.join(get_group_id(args)), min_los=args.los_min,
                       max_los=args.los_max, young_age=','.join(set([str(i) for i in range(args.age_min)])))
        patient = gcp2df(query)

    else:
        query = \
            """
            SELECT i.patientunitstayid, i.gender, i.age, i.ethnicity,  
                    CASE WHEN lower(i.hospitaldischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.hospitaldischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS hosp_mort,
                    ROUND(i.unitdischargeoffset/60) AS icu_los_hours, i.hospitaladmitoffset, i.hospitaldischargeoffset,
                   i.unitdischargeoffset, i.hospitaladmitsource, i.unitdischargelocation, 
                   CASE WHEN lower(i.unitdischargestatus) like '%alive%' THEN 0
                        WHEN lower(i.unitdischargestatus) like '%expired%' THEN 1
                        ELSE NULL END AS icu_mort, i.hospitaldischargeyear, i.hospitalid      
            From physionet-data.eicu_crd.patient i
            WHERE ROUND(i.unitdischargeoffset/60) Between {min_los} and {max_los} 
            AND age not in ({young_age})
            """.format(min_los=args.los_min, max_los=args.los_max, young_age=','.join(set([str(i) for i in range(args.age_min)])))
        patient = gcp2df(query)

    patient['unitadmitoffset'] = 0
    icuids_to_keep = patient['patientunitstayid']
    icuids_to_keep = set([str(s) for s in icuids_to_keep])
    patient.set_index('patientunitstayid', inplace=True)
    patient['max_hours'] = (patient['unitdischargeoffset'] - patient['unitadmitoffset']).apply(to_hours)
    missing_hours_fill = range_unnest(patient, 'max_hours', out_col_name='hours_in', reset_index=True)
    missing_hours_fill['tmp'] = np.NaN
    fill_df = patient.reset_index()[ID_COLS].join(missing_hours_fill.set_index('patientunitstayid'),
                                                  on='patientunitstayid')
    fill_df.set_index(ID_COLS + ['hours_in'], inplace=True)

    # Dynamic Table
    query = """
    with vw0 as
    (
      select
          patientunitstayid
        , labname
        , labresultoffset
        , labresultrevisedoffset
      from physionet-data.eicu_crd.lab
      where labname in
      (
            'paO2'
          , 'paCO2'
          , 'pH'
          , 'FiO2'
          , 'anion gap'
          , 'Base Excess'
          , 'PEEP'
      )
      group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
      having count(distinct labresult)<=1
    )
    -- get the last lab to be revised
    , vw1 as
    (
      select
          lab.patientunitstayid
        , lab.labname
        , lab.labresultoffset
        , lab.labresultrevisedoffset
        , lab.labresult
        , ROW_NUMBER() OVER
            (
              PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
              ORDER BY lab.labresultrevisedoffset DESC
            ) as rn
      from physionet-data.eicu_crd.lab
      inner join vw0
        ON  lab.patientunitstayid = vw0.patientunitstayid
        AND lab.labname = vw0.labname
        AND lab.labresultoffset = vw0.labresultoffset
        AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
      WHERE
         (lab.labname = 'paO2' and lab.labresult > 0 and lab.labresult <= 770)
      OR (lab.labname = 'paCO2' and lab.labresult > 0 and lab.labresult <= 220)
      OR (lab.labname = 'pH' and lab.labresult >= 6.3 and lab.labresult <= 10)
      OR (lab.labname = 'FiO2' and lab.labresult >= 0.2 and lab.labresult <= 1.0)
      -- we will fix fio2 units later
      OR (lab.labname = 'FiO2' and lab.labresult >= 20 and lab.labresult <= 100)
      OR (lab.labname = 'anion gap' and lab.labresult >= 0 and lab.labresult <= 55)
      OR (lab.labname = 'Base Excess' and lab.labresult >= -100 and lab.labresult <= 100)
      OR (lab.labname = 'PEEP' and lab.labresult >= 0 and lab.labresult <= 30)
    )
    select
        patientunitstayid
      , labresultoffset as chartoffset
      -- the aggregate (max()) only ever applies to 1 value due to the where clause
      , MAX(case
            when labname != 'FiO2' then null
            when labresult >= 20 then labresult/100.0
          else labresult end) as fio2
      , MAX(case when labname = 'paO2' then labresult else null end) as pao2
      , MAX(case when labname = 'paCO2' then labresult else null end) as paco2
      , MAX(case when labname = 'pH' then labresult else null end) as pH
      , MAX(case when labname = 'anion gap' then labresult else null end) as aniongap
      , MAX(case when labname = 'Base Deficit' then labresult else null end) as basedeficit
      , MAX(case when labname = 'Base Excess' then labresult else null end) as baseexcess
      , MAX(case when labname = 'PEEP' then labresult else null end) as peep
    from vw1
    where rn = 1
    and patientunitstayid in ({icuids})
    and labresultoffset >=0
    group by patientunitstayid, labresultoffset
    order by patientunitstayid, labresultoffset
    """.format(icuids=','.join(icuids_to_keep))
    bg = gcp2df(query)
    bg = fill_query(bg, fill_df)

    query = """
    with vw0 as
    (
      select
          patientunitstayid
        , labname
        , labresultoffset
        , labresultrevisedoffset
      from physionet-data.eicu_crd.lab
      where labname in
      (
          'albumin'
        , 'total bilirubin'
        , 'BUN'
        , 'calcium'
        , 'chloride'
        , 'creatinine'
        , 'bedside glucose', 'glucose'
        , 'bicarbonate' -- HCO3
        , 'Total CO2'
        , 'Hct'
        , 'Hgb'
        , 'PT - INR'
        , 'PTT'
        , 'lactate'
        , 'platelets x 1000'
        , 'potassium'
        , 'sodium'
        -- cbc related 
        , 'WBC x 1000'
        , '-bands'
        , '-basos'
        , '-eos'
        , '-lymphs'
        , '-monos'
        , '-polys'
        -- Liver enzymes
        , 'ALT (SGPT)'
        , 'AST (SGOT)'
        , 'alkaline phos.'
        -- Other 
        , 'troponin - T'
        , 'CPK-MB'
        , 'total protein'
        , 'fibrinogen'
        , 'PT'
        , 'MCH'
        , 'MCHC'
        , 'MCV'
        , 'RBC'
        , 'RDW'
        , 'amylase'
        , 'CPK'
        , 'CRP'
      )
      group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
      having count(distinct labresult)<=1
    )
    -- get the last lab to be revised
    , vw1 as
    (
      select
          lab.patientunitstayid
        , lab.labname
        , lab.labresultoffset
        , lab.labresultrevisedoffset
        , lab.labresult
        , ROW_NUMBER() OVER
            (
              PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
              ORDER BY lab.labresultrevisedoffset DESC
            ) as rn
      from physionet-data.eicu_crd.lab
      inner join vw0
        ON  lab.patientunitstayid = vw0.patientunitstayid
        AND lab.labname = vw0.labname
        AND lab.labresultoffset = vw0.labresultoffset
        AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
      -- only valid lab values
      WHERE
           (lab.labname = 'albumin' and lab.labresult > 0 and lab.labresult <=60)
        OR (lab.labname = 'total bilirubin' and lab.labresult > 0 and lab.labresult <= 66)
        OR (lab.labname = 'BUN' and lab.labresult > 0 and lab.labresult <= 300)
        OR (lab.labname = 'calcium' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'chloride' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'creatinine' and lab.labresult >0 and lab.labresult <= 66)
        OR (lab.labname in ('bedside glucose', 'glucose') and lab.labresult > 0 and lab.labresult <= 2200)
        OR (lab.labname = 'bicarbonate' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'Total CO2' and lab.labresult > 0 and lab.labresult <= 9999)
        -- will convert hct unit to fraction later
        OR (lab.labname = 'Hct' and lab.labresult > 0 and lab.labresult <= 100)
        OR (lab.labname = 'Hgb' and lab.labresult > 0 and lab.labresult <= 9999)
        OR (lab.labname = 'PT - INR' and lab.labresult > 0 and lab.labresult <= 15)
        OR (lab.labname = 'lactate' and lab.labresult > 0 and lab.labresult <= 33)
        OR (lab.labname = 'platelets x 1000' and lab.labresult >  0 and lab.labresult <= 9999)
        OR (lab.labname = 'potassium' and lab.labresult > 0 and lab.labresult <= 15)
        OR (lab.labname = 'PTT' and lab.labresult >  0 and lab.labresult <=9999)
        OR (lab.labname = 'sodium' and lab.labresult > 0 and lab.labresult <= 250)
        OR (lab.labname = 'WBC x 1000' and lab.labresult > 0 and lab.labresult <= 1100)
        OR (lab.labname = '-bands' and lab.labresult > 0 and lab.labresult <= 100)
        OR (lab.labname = '-basos' and lab.labresult > 0)
        OR (lab.labname = '-eos' and lab.labresult > 0)
        OR (lab.labname = '-lymphs' and lab.labresult > 0)
        OR (lab.labname = '-monos' and lab.labresult > 0)
        OR (lab.labname = '-polys' and lab.labresult > 0)
        OR (lab.labname = 'ALT (SGPT)' and lab.labresult > 0)
        OR (lab.labname = 'AST (SGOT)' and lab.labresult > 0)
        OR (lab.labname = 'alkaline phos.' and lab.labresult > 0)
        OR (lab.labname = 'troponin - T' and lab.labresult > 0)
        OR (lab.labname = 'CPK-MB' and lab.labresult > 0)
        OR (lab.labname = 'total protein' and lab.labresult > 0)
        OR (lab.labname = 'fibrinogen' and lab.labresult > 0)
        OR (lab.labname = 'PT' and lab.labresult > 0)
        OR (lab.labname = 'MCH' and lab.labresult > 0)
        OR (lab.labname = 'MCHC' and lab.labresult > 0)
        OR (lab.labname = 'MCV' and lab.labresult > 0)
        OR (lab.labname = 'RBC' and lab.labresult > 0)
        OR (lab.labname = 'RDW' and lab.labresult > 0)
        OR (lab.labname = 'amylase' and lab.labresult > 0)
        OR (lab.labname = 'CPK' and lab.labresult > 0)
        OR (lab.labname = 'CRP' and lab.labresult > 0)
    )
    select
        patientunitstayid
      , labresultoffset as chartoffset
      , MAX(case when labname = 'albumin' then labresult else null end) as albumin
      , MAX(case when labname = 'total bilirubin' then labresult else null end) as bilirubin
      , MAX(case when labname = 'BUN' then labresult else null end) as BUN
      , MAX(case when labname = 'calcium' then labresult else null end) as calcium
      , MAX(case when labname = 'chloride' then labresult else null end) as chloride
      , MAX(case when labname = 'creatinine' then labresult else null end) as creatinine
      , MAX(case when labname in ('bedside glucose', 'glucose') then labresult else null end) as glucose
      , MAX(case when labname = 'bicarbonate' then labresult else null end) as bicarbonate
      , MAX(case when labname = 'Total CO2' then labresult else null end) as TotalCO2
      , MAX(case when labname = 'Hct' then labresult else null end) as hematocrit
      , MAX(case when labname = 'Hgb' then labresult else null end) as hemoglobin
      , MAX(case when labname = 'PT - INR' then labresult else null end) as INR
      , MAX(case when labname = 'lactate' then labresult else null end) as lactate
      , MAX(case when labname = 'platelets x 1000' then labresult else null end) as platelets
      , MAX(case when labname = 'potassium' then labresult else null end) as potassium
      , MAX(case when labname = 'PTT' then labresult else null end) as ptt
      , MAX(case when labname = 'sodium' then labresult else null end) as sodium
      , MAX(case when labname = 'WBC x 1000' then labresult else null end) as wbc
      , MAX(case when labname = '-bands' then labresult else null end) as bands
      , MAX(case when labname = '-basos' then labresult else null end) as basos
      , MAX(case when labname = '-eos' then labresult else null end) as eos
      , MAX(case when labname = '-lymphs' then labresult else null end) as lymphs
      , MAX(case when labname = '-monos' then labresult else null end) as monos
      , MAX(case when labname = '-polys' then labresult else null end) as polys
      , MAX(case when labname = 'ALT (SGPT)' then labresult else null end) as alt
      , MAX(case when labname = 'AST (SGOT)' then labresult else null end) as ast
      , MAX(case when labname = 'alkaline phos.' then labresult else null end) as alp
      , MAX(case when labname = 'troponin - T' then labresult else null end) as troponin_t
      , MAX(case when labname = 'CPK-MB' then labresult else null end) as cpk_mb
      , MAX(case when labname = 'total protein' then labresult else null end) as total_protein
      , MAX(case when labname = 'fibrinogen' then labresult else null end) as fibrinogen
      , MAX(case when labname = 'PT' then labresult else null end) as pt
      , MAX(case when labname = 'MCH' then labresult else null end) as mch
      , MAX(case when labname = 'MCHC' then labresult else null end) as mchc
      , MAX(case when labname = 'MCV' then labresult else null end) as mcv
      , MAX(case when labname = 'RBC' then labresult else null end) as rbc
      , MAX(case when labname = 'RDW' then labresult else null end) as rdw
      , MAX(case when labname = 'amylase' then labresult else null end) as amylase
      , MAX(case when labname = 'CPK' then labresult else null end) as cpk
      , MAX(case when labname = 'CRP' then labresult else null end) as crp
    from vw1
    where rn = 1
    and patientunitstayid in ({icuids})
    and labresultoffset >=0 
    group by patientunitstayid, labresultoffset
    order by patientunitstayid, labresultoffset
    """.format(icuids=','.join(icuids_to_keep))
    lab = gcp2df(query)
    lab = fill_query(lab, fill_df)

    query = """
    with nc as
    (
    select
        patientunitstayid
      , nursingchartoffset
      , nursingchartentryoffset
      , case
          when nursingchartcelltypevallabel = 'Heart Rate'
           and nursingchartcelltypevalname = 'Heart Rate'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as heartrate
      , case
          when nursingchartcelltypevallabel = 'Respiratory Rate'
           and nursingchartcelltypevalname = 'Respiratory Rate'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as RespiratoryRate
      , case
          when nursingchartcelltypevallabel = 'O2 Saturation'
           and nursingchartcelltypevalname = 'O2 Saturation'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as o2saturation
      , case
          when nursingchartcelltypevallabel = 'Non-Invasive BP'
           and nursingchartcelltypevalname = 'Non-Invasive BP Systolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as nibp_systolic
      , case
          when nursingchartcelltypevallabel = 'Non-Invasive BP'
           and nursingchartcelltypevalname = 'Non-Invasive BP Diastolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as nibp_diastolic
      , case
          when nursingchartcelltypevallabel = 'Non-Invasive BP'
           and nursingchartcelltypevalname = 'Non-Invasive BP Mean'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as nibp_mean
      , case
          when nursingchartcelltypevallabel = 'Temperature'
           and nursingchartcelltypevalname = 'Temperature (C)'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as temperature
      --, case
      --    when nursingchartcelltypevallabel = 'Temperature'
      --     and nursingchartcelltypevalname = 'Temperature Location'
      --        then nursingchartvalue
      --    else null end
      --  as TemperatureLocation
      , case
          when nursingchartcelltypevallabel = 'Invasive BP'
           and nursingchartcelltypevalname = 'Invasive BP Systolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as ibp_systolic
      , case
          when nursingchartcelltypevallabel = 'Invasive BP'
           and nursingchartcelltypevalname = 'Invasive BP Diastolic'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as ibp_diastolic
      , case
          when nursingchartcelltypevallabel = 'Invasive BP'
           and nursingchartcelltypevalname = 'Invasive BP Mean'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          -- other map fields
          when nursingchartcelltypevallabel = 'MAP (mmHg)'
           and nursingchartcelltypevalname = 'Value'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          when nursingchartcelltypevallabel = 'Arterial Line MAP (mmHg)'
           and nursingchartcelltypevalname = 'Value'
           and REGEXP_CONTAINS(nursingchartvalue, r'^[-]?[0-9]+[.]?[0-9]*$')
           and nursingchartvalue not in ('-','.')
              then cast(nursingchartvalue as numeric)
          else null end
        as ibp_mean
      from physionet-data.eicu_crd.nursecharting
      -- speed up by only looking at a subset of charted data
      where nursingchartcelltypecat in
      (
        'Vital Signs','Scores','Other Vital Signs and Infusions'
      )
    )
    select
      patientunitstayid
    , nursingchartoffset as chartoffset
    , nursingchartentryoffset as entryoffset
    , avg(case when heartrate > 0 and heartrate <= 390 then heartrate else null end) as heartrate
    , avg(case when RespiratoryRate >= 0 and RespiratoryRate <= 330 then RespiratoryRate else null end) as RespiratoryRate
    , avg(case when o2saturation >= 0 and o2saturation <= 100 then o2saturation else null end) as spo2
    , avg(case when nibp_systolic > 0 and nibp_systolic <= 375 then nibp_systolic else null end) as nibp_systolic
    , avg(case when nibp_diastolic > 0 and nibp_diastolic <= 375 then nibp_diastolic else null end) as nibp_diastolic
    , avg(case when nibp_mean > 0 and nibp_mean <= 375 then nibp_mean else null end) as nibp_mean
    , avg(case when temperature >= 14.2 and temperature <= 47 then temperature else null end) as temperature
    --, max(temperaturelocation) as temperaturelocation
    , avg(case when ibp_systolic > 0 and ibp_systolic <= 375 then ibp_systolic else null end) as ibp_systolic
    , avg(case when ibp_diastolic > 0 and ibp_diastolic <= 375 then ibp_diastolic else null end) as ibp_diastolic
    , avg(case when ibp_mean > 0 and ibp_mean <= 375 then ibp_mean else null end) as ibp_mean
    from nc
    WHERE (heartrate IS NOT NULL
    OR RespiratoryRate IS NOT NULL
    OR o2saturation IS NOT NULL
    OR nibp_systolic IS NOT NULL
    OR nibp_diastolic IS NOT NULL
    OR nibp_mean IS NOT NULL
    OR temperature IS NOT NULL
    --OR temperaturelocation IS NOT NULL
    OR ibp_systolic IS NOT NULL
    OR ibp_diastolic IS NOT NULL
    OR ibp_mean IS NOT NULL)
    AND patientunitstayid in ({icuids})
    AND nursingchartoffset >=0 
    group by patientunitstayid, nursingchartoffset, nursingchartentryoffset
    order by patientunitstayid, nursingchartoffset, nursingchartentryoffset
    """.format(icuids=','.join(icuids_to_keep))
    vital = gcp2df(query)

    vital.drop('entryoffset', axis=1, inplace=True)
    vital = fill_query(vital, fill_df)

    # microlab
    query = """
    SELECT ml.patientunitstayid, ml.culturetakenoffset
        , case
            when ml.culturesite = 'Blood, Venipuncture' then 'culturesite0'
            when ml.culturesite in ('Urine, Catheter Specimen', 'Urine, Voided Specimen') then 'culturesite1'
            when ml.culturesite = 'Nasopharynx' then 'culturesite2'
            when ml.culturesite = 'Stool' then 'culturesite3'
            when ml.culturesite in ('Sputum, Tracheal Specimen', 'Sputum, Expectorated') then 'culturesite4'
            when ml.culturesite = 'CSF' then 'culturesite8'
            when ml.culturesite = 'Peritoneal Fluid' then 'culturesite9'
            when ml.culturesite = 'Bronchial Lavage' then 'culturesite11'
            when ml.culturesite = 'Rectal Swab' then 'culturesite12'
            when ml.culturesite in ('Other', 'Wound, Decubitus', 'Pleural Fluid', 
                  'Bile', 'Skin', 'Wound, Surgical', 'Wound, Drainage Fluid', 'Blood, Central Line', 'Abscess')
                  then 'culturesite13'
            else null end as culturesite
        , case
            when ml.organism = 'no growth' then 0
            when ml.organism != ""  then 1
            else null end as positive
        , case 
            when ml.antibiotic != ""  then 1
            else null end as screen
        , case 
            when ml.sensitivitylevel = 'Sensitive' then 1
            when ml.sensitivitylevel = 'Resistant' then 0 
            else null end as has_sensitivity
    FROM physionet-data.eicu_crd.microlab ml
    WHERE ml.patientunitstayid in ({icuids})
    AND ml.culturetakenoffset >=0

    """.format(icuids=','.join(icuids_to_keep))
    microlab = gcp2df(query, fill_df)

    microlab['hours_in'] = microlab['culturetakenoffset'].floordiv(60)
    microlab.drop(columns=['culturetakenoffset'], inplace=True)
    microlab.reset_index(inplace=True)
    microlab = microlab.groupby(ID_COLS + ['hours_in']).agg(['last'])
    microlab = microlab.reindex(fill_df.index)

    query = """
    SELECT gc.patientunitstayid	, gc.chartoffset, gc.gcs
    FROM physionet-data.eicu_crd_derived.pivoted_gcs gc
    WHERE gc.patientunitstayid in ({icuids})
    and gc.chartoffset >=0

    """.format(icuids=','.join(icuids_to_keep))
    gcs = gcp2df(query)
    gcs = fill_query(gcs, fill_df)

    # uo weight cvp
    query = """
    SELECT uo.patientunitstayid, uo.chartoffset, uo.urineoutput
    FROM physionet-data.eicu_crd_derived.pivoted_uo uo
    WHERE uo.patientunitstayid in ({icuids})
    and uo.chartoffset >=0

    """.format(icuids=','.join(icuids_to_keep))
    uo = gcp2df(query)
    uo = fill_query(uo, fill_df)

    # weight cvp
    query = """
    SELECT wg.patientunitstayid, wg.chartoffset, wg.weight
    FROM physionet-data.eicu_crd_derived.pivoted_weight wg
    WHERE wg.patientunitstayid in ({icuids})
    and wg.chartoffset >=0

    """.format(icuids=','.join(icuids_to_keep))
    weight = gcp2df(query)
    weight = fill_query(weight, fill_df)

    # weight cvp
    query = """
    SELECT vp.patientunitstayid, vp.observationoffset, vp.cvp
    FROM physionet-data.eicu_crd.vitalperiodic vp
    WHERE vp.patientunitstayid in ({icuids})
    and vp.observationoffset >=0
    """.format(icuids=','.join(icuids_to_keep))
    cvp = gcp2df(query)
    cvp = fill_query(cvp, fill_df, time='observationoffset')

    # concat all
    vital = bg.join([lab, vital, gcs, uo, weight, cvp, microlab])

    del bg, lab, gcs, uo, weight, cvp, microlab

    # prepare some make up
    # not perfect it affects percentage calculation
    query = """
    with vw0 as
    (
      select
          patientunitstayid
        , labname
        , labresultoffset
        , labresultrevisedoffset
      from physionet-data.eicu_crd.lab
      where labname in
      ('urinary creatinine', 'magnesium',  'phosphate', "WBC's in urine", '24 h urine protein'
      )
      group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
      having count(distinct labresult)<=1
    )
    -- get the last lab to be revised
    , vw1 as
    (
      select
          lab.patientunitstayid
        , lab.labname
        , lab.labresultoffset
        , lab.labresultrevisedoffset
        , lab.labresult
        , ROW_NUMBER() OVER
            (
              PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
              ORDER BY lab.labresultrevisedoffset DESC
            ) as rn
      from physionet-data.eicu_crd.lab
      inner join vw0
        ON  lab.patientunitstayid = vw0.patientunitstayid
        AND lab.labname = vw0.labname
        AND lab.labresultoffset = vw0.labresultoffset
        AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
      -- only valid lab values
      WHERE
           (lab.labname = 'urinary creatinine' and lab.labresult > 0 and lab.labresult <= 650) -- based on mimic 
        OR (lab.labname = 'magnesium' and lab.labresult > 0 and lab.labresult <= 22)
        OR (lab.labname = 'phosphate' and lab.labresult > 0 and lab.labresult <= 22)
        OR (lab.labname = "WBC's in urine" and lab.labresult > 0 and lab.labresult <= 750) -- based on mimic
        OR (lab.labname = '24 h urine protein'and lab.labresult > 0) -- no need 
    )
    select
        patientunitstayid
      , labresultoffset as chartoffset
      , MAX(case when labname = 'urinary creatinine' then labresult else null end) as urine_creat
      , MAX(case when labname = 'magnesium' then labresult else null end) as magnesium
      , MAX(case when labname = 'phosphate' then labresult else null end) as phosphate
      , MAX(case when labname = "WBC's in urine" then labresult else null end) as wbc_urine
      , MAX(case when labname = '24 h urine protein' then labresult else null end) as urine_prot
    from vw1
    where rn = 1
    and patientunitstayid in ({icuids})
    and labresultoffset >=0
    group by patientunitstayid, labresultoffset
    order by patientunitstayid, labresultoffset
    """.format(icuids=','.join(icuids_to_keep))
    labmakeup = gcp2df(query)
    labmakeup = fill_query(labmakeup, fill_df)

    query = """
    SELECT rc.patientunitstayid, 
            rc.respchartoffset as chartoffset, cast(rc.respchartvalue as FLOAT64) as tidal_vol_obs
    FROM physionet-data.eicu_crd.respiratorycharting rc
    WHERE rc.respchartvaluelabel = 'Tidal Volume Observed (VT)'
    AND patientunitstayid in ({icuids})
    AND respchartoffset >=0
    """.format(icuids=','.join(icuids_to_keep))
    tidal_vol_obs = gcp2df(query)
    tidal_vol_obs = fill_query(tidal_vol_obs, fill_df)

    vital = vital.join([labmakeup, tidal_vol_obs])
    del labmakeup, tidal_vol_obs

    idx = pd.IndexSlice
    vital.loc[:, idx[:, 'count']] = vital.loc[:, idx[:, 'count']].fillna(0)

    original = vital.loc[:, idx['ibp_systolic', ['mean', 'count']]].copy(deep=True)
    makeups = vital.loc[:, idx['nibp_systolic', ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    vital.loc[:, idx['ibp_systolic', ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values

    original = vital.loc[:, idx['ibp_diastolic', ['mean', 'count']]].copy(deep=True)
    makeups = vital.loc[:, idx['nibp_diastolic', ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    vital.loc[:, idx['ibp_diastolic', ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values

    original = vital.loc[:, idx['ibp_mean', ['mean', 'count']]].copy(deep=True)
    makeups = vital.loc[:, idx['nibp_mean', ['mean', 'count']]].copy(deep=True)
    filled = combine_cols(makeups, original)
    vital.loc[:, idx['ibp_mean', ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values

    # drop 'basedeficit' 124 + 5-->  122+4
    vital.drop('basedeficit', axis=1, level=0, inplace=True)
    vital.drop('index', axis=1, level=0, inplace=True)

    vital = pd.get_dummies(vital)
    # screen and positive culture needs impute, they are last columns but with float data type
    vital[('positive', 'mask')] = (~vital[('positive', 'last')].isnull()).astype(float)
    vital[('screen', 'mask')] = (~vital[('screen', 'last')].isnull()).astype(float)
    vital[('has_sensitivity', 'mask')] = (~vital[('has_sensitivity', 'last')].isnull()).astype(float)

    # make empty columns
    columns_to_make = ['calcium_0', 'atypical_lymphocytes', 'immature_granulocytes', 'metamyelocytes', 'nrbc',
                       'ntprobnp', 'bilirubin_direct', 'bilirubin_indirect', 'ggt', 'ld_ldh',
                       'Peak inspiratory pressure', 'Plateau Pressure',
                       'Positive end-expiratory pressure Set', 'Red blood cell count urine', 'pH urine']
    for col in columns_to_make:
        vital[(col, 'mean')] = fill_df.values
        vital[(col, 'count')] = 0

    # other columsn
    empty_culture = ["('culturesite', 'last')_culturesite10", "('culturesite', 'last')_culturesite5",
                     "('culturesite', 'last')_culturesite6", "('culturesite', 'last')_culturesite7"]
    for col in empty_culture:
        vital[col] = 0

    # organize columns
    # combine glucose_1 to glucose in mimic, very similair to glucose
    # combine blood urea nitrogen to bun # remember means and stds
    # drop abs cells
    # concate with makeup columns
    # col = X_encode.columns.tolist()
    # 138 = 61*2 + 6 + 10
    col = ['spo2', 'pao2', 'paco2', 'fio2', 'pH', 'baseexcess',
           'bicarbonate', 'TotalCO2', 'hematocrit', 'hemoglobin', 'chloride', 'calcium_0',
           'temperature', 'potassium', 'sodium', 'lactate', 'glucose', 'heartrate',
           'ibp_systolic', 'ibp_diastolic', 'ibp_mean', 'nibp_systolic', 'nibp_diastolic', 'nibp_mean',
           'RespiratoryRate', 'wbc', 'basos', 'eos', 'lymphs', 'monos', 'polys',
           'atypical_lymphocytes', 'bands', 'immature_granulocytes', 'metamyelocytes', 'nrbc',
           'troponin_t', 'cpk_mb', 'ntprobnp', 'albumin', 'total_protein', 'aniongap',
           'BUN', 'calcium', 'creatinine', 'fibrinogen', 'INR', 'pt', 'ptt', 'mch',
           'mchc', 'mcv', 'platelets', 'rbc', 'rdw', 'alt', 'alp', 'ast', 'amylase', 'bilirubin',
           'bilirubin_direct', 'bilirubin_indirect', 'cpk', 'ggt', 'ld_ldh', 'gcs', 'crp',
           'weight', 'urineoutput', 'cvp', 'urine_creat', 'magnesium', 'Peak inspiratory pressure',
           'phosphate', 'Plateau Pressure', 'peep', 'Positive end-expiratory pressure Set',
           'Red blood cell count urine',
           'tidal_vol_obs', 'urine_prot', 'wbc_urine', 'pH urine', 'positive',
           'screen', 'has_sensitivity',
           "('culturesite', 'last')_culturesite0", "('culturesite', 'last')_culturesite1",
           "('culturesite', 'last')_culturesite10",
           "('culturesite', 'last')_culturesite11",
           "('culturesite', 'last')_culturesite12", "('culturesite', 'last')_culturesite13",
           "('culturesite', 'last')_culturesite2", "('culturesite', 'last')_culturesite3",
           "('culturesite', 'last')_culturesite4",
           "('culturesite', 'last')_culturesite5", "('culturesite', 'last')_culturesite6",
           "('culturesite', 'last')_culturesite7",
           "('culturesite', 'last')_culturesite8", "('culturesite', 'last')_culturesite9"]

    # generate final col
    breakpoint1 = col.index('positive')
    breakpoint2 = col.index("('culturesite', 'last')_culturesite0")
    col_ready = []
    for i in range(breakpoint1):
        col_ready.append((col[i], 'mean'))
        col_ready.append((col[i], 'count'))
    for i in range(breakpoint1, breakpoint2):
        col_ready.append((col[i], 'last'))
        col_ready.append((col[i], 'mask'))
    for i in range(breakpoint2, len(col)):
        col_ready.append(col[i])

    vital = vital[col_ready]

    # Intervention table
    query = \
        """
        with 
        ventall as
        (
            SELECT Distinct i.patientunitstayid, i.priorventstartoffset, i.priorventendoffset	
            From physionet-data.eicu_crd.respiratorycare i
            WHERE i.priorventstartoffset >0 or i.priorventendoffset	>0
            Order by patientunitstayid
        )
    
        SELECT vt.patientunitstayid, FLOOR(GREATEST(vt.priorventstartoffset, 0)/60) as starttime, 
            FLOOR(LEAST(vt.priorventendoffset, i.unitdischargeoffset)/60) as endtime,
           FLOOR((i.unitdischargeoffset - i.unitadmitoffset)/60) as max_hours
        FROM ventall vt
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = vt.patientunitstayid
        WHERE  vt.priorventstartoffset is not null 
        AND vt.priorventendoffset is not null
        AND vt.patientunitstayid in ({icuids})
        """.format(icuids=','.join(icuids_to_keep))
    vent = gcp2df(query)

    vent_data = process_inv(vent, 'vent')
    ids_with = vent_data['patientunitstayid']
    ids_with = set(map(int, ids_with))
    ids_all = set(map(int, icuids_to_keep))
    ids_without = (ids_all - ids_with)

    # patient.set_index('patientunitstayid', inplace=True)
    icustay_timediff_tmp = patient['unitdischargeoffset'] - patient['unitadmitoffset']
    icustay_timediff = pd.Series([timediff // 60
                                  for timediff in icustay_timediff_tmp], index=patient.index.values)
    # Create a new fake dataframe with blanks on all vent entries
    out_data = fill_df.copy(deep=True)
    out_data = out_data.reset_index()
    out_data = out_data.set_index('patientunitstayid')
    out_data = out_data.iloc[out_data.index.isin(ids_without)]
    out_data = out_data.reset_index()
    out_data = out_data[['patientunitstayid']]
    out_data['max_hours'] = out_data['patientunitstayid'].map(icustay_timediff)

    # Create all 0 column for vent
    out_data = out_data.groupby('patientunitstayid')
    out_data = out_data.apply(add_blank_indicators_e)
    out_data.rename(columns={'on': 'vent'}, inplace=True)

    out_data = out_data.reset_index()
    intervention = pd.concat([vent_data[['patientunitstayid', 'hours_in', 'vent']],
                              out_data[['patientunitstayid', 'hours_in', 'vent']]],
                             axis=0)

    # pivoted_med
    column_names = ['dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine', 'vasopressin', 'dobutamine',
                    'milrinone', 'heparin']

    for c in column_names:
        query = \
            """
            SELECT pm.patientunitstayid, FLOOR(GREATEST(pm.drugorderoffset, 0)/60) as starttime, FLOOR(LEAST(pm.drugstopoffset, i.unitdischargeoffset)/60) as endtime, 
                pm.{drug_name}, 
                FLOOR((i.unitdischargeoffset - i.unitadmitoffset)/60) as max_hours
            FROM physionet-data.eicu_crd_derived.pivoted_med pm
            INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = pm.patientunitstayid
            WHERE pm.{drug_name} = 1 
            AND pm.patientunitstayid in ({icuids}) 
            AND pm.drugorderoffset is not null 
            AND pm.drugstopoffset is not null
            """.format(drug_name=c, icuids=','.join(icuids_to_keep))
        med = gcp2df(query)
        # 'epinephrine',  'dopamine', 'norepinephrine', 'phenylephrine', \
        #    'vasopressin', 'dobutamine', 'milrinone',  'heparin',
        med = process_inv(med, c)
        intervention = intervention.merge(
            med[['patientunitstayid', 'hours_in', c]],
            on=['patientunitstayid', 'hours_in'],
            how='left'
        )

    # antibiotics
    query = \
        """
        SELECT md.patientunitstayid, FLOOR(GREATEST(md.drugstartoffset, 0)/60) as starttime
            , FLOOR(LEAST(md.drugstopoffset, i.unitdischargeoffset)/60) as endtime
            , FLOOR((i.unitdischargeoffset - i.unitadmitoffset)/60) as max_hours
        FROM physionet-data.eicu_crd.medication md
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = md.patientunitstayid
        WHERE (REGEXP_CONTAINS(lower(drugname), r"^.*adoxa.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ala-tet.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*alodox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amikacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amikin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amoxicill.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*amphotericin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*anidulafungin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ancef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*clavulanate.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ampicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*augmentin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*avelox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*avidoxy.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*azactam.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*azithromycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*aztreonam.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*axetil.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bactocill.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bactrim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bactroban.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bethkis.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*biaxin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*bicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cayston.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefazolin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cedax.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefoxitin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftazidime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefaclor.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefadroxil.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefdinir.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefditoren.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefepime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefotan.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefotetan.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefotaxime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftaroline.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefpodoxime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefpirome.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefprozil.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftibuten.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ceftriaxone.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cefuroxime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cephalexin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cephalothin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cephapririn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*chloramphenicol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cipro.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ciprofloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*claforan.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*clarithromycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cleocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*clindamycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*cubicin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*dicloxacillin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*dirithromycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*doryx.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*doxycy.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*duricef.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*dynacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ery-tab.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*eryped.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*eryc.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*erythrocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*erythromycin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*factive.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*flagyl.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*fortaz.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*furadantin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*garamycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*gentamicin.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*kanamycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*keflex.*$") 
          OR REGEXP_CONTAINS(lower(drugname), r"^.*kefzol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ketek.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*levaquin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*levofloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*lincocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*linezolid.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*macrobid.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*macrodantin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*maxipime.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*mefoxin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*metronidazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*meropenem.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*methicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*minocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*minocycline.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*monodox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*monurol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*morgidox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*moxatag.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*moxifloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*mupirocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*myrac.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*nafcillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*neomycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*nicazel doxy 30.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*nitrofurantoin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*norfloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*noroxin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ocudox.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*ofloxacin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*omnicef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*oracea.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*oraxyl.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*oxacillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pc pen vk.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pce dispertab.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*panixine.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pediazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*penicillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*periostat.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*pfizerpen.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*piperacillin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tazobactam.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*primsol.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*proquin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*raniclor.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*rifadin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*rifampin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*rocephin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*smz-tmp.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*septra.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*septra ds.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*septra.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*solodyn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*spectracef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*streptomycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfadiazine.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfamethoxazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*trimethoprim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfatrim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*sulfisoxazole.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*suprax.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*synercid.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tazicef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tetracycline.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*timentin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*tobramycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*trimethoprim.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*unasyn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vancocin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vancomycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vantin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vibativ.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vibra-tabs.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*vibramycin.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zinacef.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zithromax.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zosyn.*$")
          OR REGEXP_CONTAINS(lower(drugname), r"^.*zyvox.*$")
          )
        AND md.drugordercancelled = 'No'
        AND md.patientunitstayid in ({icuids}) 
        AND md.drugstartoffset is not null 
        AND md.drugstopoffset is not null
        """.format(icuids=','.join(icuids_to_keep))

    anti = gcp2df(query)

    anti = process_inv(anti, 'antib')
    intervention = intervention.merge(
        anti[['patientunitstayid', 'hours_in', 'antib']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    ## crrt
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/60) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/60) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/60) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*crrt.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep))
    crrt = gcp2df(query)

    crrt = process_inv(crrt, 'crrt')
    intervention = intervention.merge(
        crrt[['patientunitstayid', 'hours_in', 'crrt']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    ##  rbc transfusion
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/60) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/60) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/60) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE (REGEXP_CONTAINS(lower(cellpath), r"^.*rbc.*$")
        OR REGEXP_CONTAINS(lower(cellpath), r"^.*red blood cell.*$"))
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep))
    rbc = gcp2df(query)

    rbc = process_inv(rbc, 'rbc')
    intervention = intervention.merge(
        rbc[['patientunitstayid', 'hours_in', 'rbc']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    ##  ffp transfusion
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/60) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/60) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/60) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE (REGEXP_CONTAINS(lower(cellpath), r"^.*plasma.*$")
        OR REGEXP_CONTAINS(lower(cellpath), r"^.*ffp.*$"))
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep))
    ffp = gcp2df(query)
    ffp = process_inv(ffp, 'ffp')
    intervention = intervention.merge(
        ffp[['patientunitstayid', 'hours_in', 'ffp']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    ##  platelets transfusion
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/60) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/60) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/60) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*platelet.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep))
    platelets = gcp2df(query)

    platelets = process_inv(platelets, 'platelets')
    intervention = intervention.merge(
        platelets[['patientunitstayid', 'hours_in', 'platelets']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    ##
    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/60) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/60) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/60) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*colloid.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep))
    colloid = gcp2df(query)
    colloid = process_inv(colloid, 'colloid')
    intervention = intervention.merge(
        colloid[['patientunitstayid', 'hours_in', 'colloid']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    query = \
        """
        SELECT io.patientunitstayid, FLOOR(GREATEST(MIN(io.intakeoutputoffset), 0)/60) as starttime
            , FLOOR(LEAST(MAX(io.intakeoutputoffset), MIN(i.unitdischargeoffset))/60) as endtime
            , FLOOR((MIN(i.unitdischargeoffset) - MIN(i.unitadmitoffset))/60) as max_hours
        FROM physionet-data.eicu_crd.intakeoutput io
        INNER JOIN physionet-data.eicu_crd_derived.icustay_detail i ON i.patientunitstayid = io.patientunitstayid
        WHERE REGEXP_CONTAINS(lower(cellpath), r"^.*crystalloid.*$")
        AND io.patientunitstayid in ({icuids}) 
        GROUP BY io.patientunitstayid
        """.format(icuids=','.join(icuids_to_keep))
    crystalloid = gcp2df(query)
    crystalloid = process_inv(crystalloid, 'crystalloid')
    intervention = intervention.merge(
        crystalloid[['patientunitstayid', 'hours_in', 'crystalloid']],
        on=['patientunitstayid', 'hours_in'],
        how='left'
    )

    # for each column, astype to int and fill na with 0
    intervention = intervention.fillna(0)
    for i in range(3, 18):
        intervention.iloc[:, i] = intervention.iloc[:, i].astype(int)

    intervention.set_index(ID_COLS + ['hours_in'], inplace=True)
    intervention.sort_index(level=['patientunitstayid', 'hours_in'], inplace=True)

    new_col = ['vent', 'antib', 'dopamine', 'epinephrine', 'norepinephrine', 'phenylephrine',
               'vasopressin', 'dobutamine', 'milrinone', 'heparin', 'crrt',
               'rbc', 'platelets', 'ffp', 'colloid', 'crystalloid']
    intervention = intervention.loc[:, new_col]
    # static query
    # commo
    query = \
        """
        SELECT ad.patientunitstayid
    
            -- Myocardial infarction
            , MAX(CASE WHEN
                SUBSTR(ad.icd9code, 1, 3) IN ('410','412')
                THEN 1 
                ELSE 0 END) AS myocardial_infarct
    
            -- Congestive heart failure
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) = '428'
                OR
                SUBSTR(ad.icd9code, 1, 6) IN ('398.91','402.01','402.11','402.91','404.01','404.03',
                                '404.11','404.13','404.91','404.93')
                OR 
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '425.4' AND '425.9'
                THEN 1 
                ELSE 0 END) AS congestive_heart_failure
    
            -- Peripheral vascular disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('440','441')
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('093.0','437.3','4471.','557.1','557.9','V43.4')
                OR
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '443.1' AND '443.9'
                THEN 1 
                ELSE 0 END) AS peripheral_vascular_disease
    
            -- Cerebrovascular disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '430' AND '438'
                OR
                SUBSTR(ad.icd9code, 1, 6) = '362.34'
                THEN 1 
                ELSE 0 END) AS cerebrovascular_disease
    
            -- Dementia
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) = '290'
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('294.1','331.2')
                THEN 1 
                ELSE 0 END) AS dementia
    
            -- Chronic pulmonary disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '490' AND '505'
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('416.8','416.9','506.4','508.1','508.8')
                THEN 1 
                ELSE 0 END) AS chronic_pulmonary_disease
    
            -- Rheumatic disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) = '725'
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('446.5','710.0','710.1','710.2','710.3',
                                                        '710.4','714.0','714.1','714.2','714.8')
                THEN 1 
                ELSE 0 END) AS rheumatic_disease
    
            -- Peptic ulcer disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('531','532','533','534')
                THEN 1 
                ELSE 0 END) AS peptic_ulcer_disease
    
            -- Mild liver disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('570','571')
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('070.6','070.9','573.3','573.4','573.8','573.9','V42.7')
                OR
                SUBSTR(ad.icd9code, 1, 6) IN ('070.22','070.23','070.32','070.33','070.44','070.54')
                THEN 1 
                ELSE 0 END) AS mild_liver_disease
    
            -- Diabetes without chronic complication
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 5) IN ('250.0','250.1','250.2','250.3','250.8','250.9') 
                THEN 1 
                ELSE 0 END) AS diabetes_without_cc
    
            -- Diabetes with chronic complication
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 5) IN ('250.4','250.5','250.6','250.7')
                THEN 1 
                ELSE 0 END) AS diabetes_with_cc
    
            -- Hemiplegia or paraplegia
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('342','343')
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('334.1','344.0','344.1','344.2',
                                                        '344.3','344.4','344.5','344.6','344.9')
                THEN 1 
                ELSE 0 END) AS paraplegia
    
            -- Renal disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('582','585','586','V56')     
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('588.0','V42.0','V45.1')  
                OR
                SUBSTR(ad.icd9code, 1, 5) IN ('583.0', '583.1','583.2','583.3','583.4','583.5', '583.6','583.7')
                OR
                SUBSTR(ad.icd9code, 1, 6) IN ('403.01','403.11','403.91','404.02','404.03','404.12','404.13','404.92','404.93')  
                THEN 1 
                ELSE 0 END) AS renal_disease
    
            -- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '140' AND '172'
                OR
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '174.0' AND '195.8'
                OR
                SUBSTR(ad.icd9code, 1, 3) BETWEEN '200' AND '208'
                OR
                SUBSTR(ad.icd9code, 1, 5) = '238.6'
                THEN 1 
                ELSE 0 END) AS malignant_cancer
    
            -- Moderate or severe liver disease
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 5) IN ('456.0','456.1','456.2')
                OR
                SUBSTR(ad.icd9code, 1, 5) BETWEEN '572.2' AND '572.8'
                THEN 1 
                ELSE 0 END) AS severe_liver_disease
    
            -- Metastatic solid tumor
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('196','197','198','199')
                THEN 1 
                ELSE 0 END) AS metastatic_solid_tumor
    
            -- AIDS/HIV
            , MAX(CASE WHEN 
                SUBSTR(ad.icd9code, 1, 3) IN ('042','043','044')
                THEN 1 
                ELSE 0 END) AS aids
    
        FROM physionet-data.eicu_crd.diagnosis ad
        WHERE ad.patientunitstayid in ({icuids})
        GROUP BY ad.patientunitstayid
        ;
        """.format(icuids=','.join(icuids_to_keep))
    commo = gcp2df(query)
    commo.set_index('patientunitstayid', inplace=True)
    static = patient.join(commo)
    static_col = static.columns.tolist()
    static_col.remove('hospitalid')
    static_col.append('hospitalid')
    static = static[static_col]

    intervention.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_inv.h5'), key='eicu_inv')
    static.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_static.h5'), key='eicu_static')
    vital.to_hdf(os.path.join(args.output_dir, 'MEEP_eICU_vital.h5'), key='eicu_vital')

    total_cols = vital.columns.tolist()
    mean_col = [i for i in total_cols if 'mean' in i]
    X_mean = vital.loc[:, mean_col]

    range_dict_high = {'calcium': 28,  'chloride': 200, 'bicarbonate': 66, 'TotalCO2': 80, 'hemoglobin': 30,
                       'platelets': 2200, 'ptt': 150, 'basos': 8,  'alp': 4000, 'ast': 22000, 'alt': 11000,
                       'troponin_t': 24, 'cpk_mb': 700, 'cpk': 10000,  'pt': 150, 'mch': 46, 'mchc': 43,
                       'mcv': 140, 'rbc': 8, 'rdw': 38, 'amylase': 2800, 'crp': 4000, 'urineoutput': 2445,
                       'weight': 550, 'urine_prot': 7500, 'cvp': 400}
    range_dict_low = {}

    for var_to_remove in range_dict_high:
        remove_outliers_h(vital, X_mean, var_to_remove, range_dict_high[var_to_remove])
    for var_to_remove in range_dict_low:
        remove_outliers_l(vital, X_mean, var_to_remove, range_dict_low[var_to_remove])

    # read_mimic col means col stds
    mimic_mean_std = pd.read_hdf(os.path.join('./Extract', 'MEEP_stats_0702.h5'), key='vital_mean_std')
    del X_mean
    # normalize
    count_col = [i for i in total_cols if 'count' in i]
    # fix fio2 column by x100
    vital.loc[:, [('fio2', 'mean')]] = vital.loc[:, [('fio2', 'mean')]] * 100
    # col_means, col_stds = vital.loc[:, mean_col].mean(axis=0), vital.loc[:, mean_col].std(axis=0)
    # first use mimic mean to normorlize
    col_means, col_stds = mimic_mean_std.loc[:, 'mean'], mimic_mean_std.loc[:, 'std']
    col_means.index = mean_col
    col_stds.index = mean_col
    vital.loc[:, mean_col] = (vital.loc[:, mean_col] - col_means) / col_stds
    icustay_means = vital.loc[:, mean_col].groupby(ID_COLS).mean()
    # impute
    vital.loc[:, mean_col] = vital.loc[:, mean_col].groupby(ID_COLS).fillna(method='ffill').groupby(ID_COLS).fillna(
        icustay_means).fillna(0)
    # 0 or 1
    vital.loc[:, count_col] = (vital.loc[:, count_col] > 0).astype(float)
    # at this satge only 3 last columns has nan values
    vital = vital.fillna(0)

    # split data
    stays_v = set(vital.index.get_level_values(0).values)
    stays_static = set(static.index.get_level_values(0).values)
    stays_int = set(intervention.index.get_level_values(0).values)
    assert stays_v == stays_static, "Stay ID pools differ!"
    assert stays_v == stays_int, "Stay ID pools differ!"
    train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
    SEED = 41
    np.random.seed(SEED)
    subjects, N = np.random.permutation(list(stays_v)), len(stays_v)
    N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
    train_stay = list(stays_v)[:N_train]
    dev_stay = list(stays_v)[N_train:N_train + N_dev]
    test_stay = list(stays_v)[N_train + N_dev:]

    [(vital_train, vital_dev, vital_test), (Y_train, Y_dev, Y_test), (static_train, static_dev, static_test)] = [
        [df[df.index.get_level_values(0).isin(s)] for s in (train_stay, dev_stay, test_stay)] \
        for df in (vital, intervention, static)]

    if args.exit_point == 'All':
        vital_train.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='vital_train')
        vital_dev.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='vital_dev')
        vital_test.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='vital_test')
        Y_train.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='inv_train')
        Y_dev.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='inv_dev')
        Y_test.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='inv_test')
        static_train.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='static_train')
        static_dev.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='static_dev')
        static_test.to_hdf('./Extract/MEEP/eICU_split_1.hdf5', key='static_test')
    return
