# Set up Google big query
from google.colab import auth
from google.cloud import bigquery

auth.authenticate_user()

import os
import numpy as np
import pandas as pd
from utils_mimic import *


def extract_mimic(args):
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project_id
    client = bigquery.Client(project=args.project_id)

    def gcp2df(sql, job_config=None):
        query = client.query(sql, job_config)
        results = query.result()
        return results.to_dataframe()

    level_to_change = 1
    ID_COLS = ['subject_id', 'hadm_id', 'stay_id']
    ITEM_COLS = ['itemid', 'label', 'LEVEL1', 'LEVEL2']

    # datatime format to hour
    to_hours = lambda x: max(0, x.days * 24 + x.seconds // 3600)

    # define our patient cohort by age, icu stay time
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
    LIMIT 1000
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
    SELECT b.subject_id, b.charttime, b.specimen, b.screen, b.positive_culture, b.has_sensitivity, i.hadm_id, i.stay_id, i.icu_intime
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
    chartitems_to_keep = pd.read_excel('/resources/chartitems_to_keep_0505.xlsx')
    lab_to_keep = pd.read_excel('/resources/labitems_to_keep_0505.xlsx')
    var_map = pd.read_csv('/resources/Chart_makeup_0505 - var_map0505.csv')
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

    chart_lab.loc[:, idx[:, ['count']]] = chart_lab.loc[:, idx[:, ['count']]].fillna(0)
    total.loc[:, idx[:, ['count']]] = total.loc[:, idx[:, ['count']]].fillna(0)

    # combine columns since they were from different itemids
    names_to_combine = [
        ['so2', 'spo2'], ['fio2', 'fio2_chartevents'], ['bicarbonate', 'bicarbonate_chem'],
        ['hematocrit', 'hematocrit_cbc'], ['hemoglobin', 'hemoglobin_cbc'], ['chloride', 'chloride_chem'],
        ['glucose', 'glucose_chem'], ['glucose', 'glucose_vital'],
        ['temperature', 'temp_vital'], ['sodium', 'sodium_chem'], ['potassium', 'potassium_chem']
    ]
    idx = pd.IndexSlice
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

    # ['alt', 'Alanine aminotransferase'], ['albumin','Albumin'], ['alp', 'Alkaline phosphate'],\
    # ['aniongap', 'Anion gap'], ['ast', 'Asparate aminotransferase'], ['bicarbonate', 'Bicarbonate'], \
    # ['bilirubin_total', 'Bilirubin'], ['bilirubin_direct','Bilirubin_direct'], ['calcium', 'Calcium ionized'],\
    # ['chloride', 'Chloride'], ['creatinine', 'Creatinine'], ['dbp', 'Diastolic blood pressure'], ['dbp_ni', 'Diastolic blood pressure'],\
    # Combine between chartlab and total table
    names_list = [
        ['alt', 'Alanine aminotransferase'], ['albumin', 'Albumin'], ['alp', 'Alkaline phosphate'], \
        ['aniongap', 'Anion gap'], ['ast', 'Asparate aminotransferase'], ['bicarbonate', 'Bicarbonate'], \
        ['bilirubin_total', 'Bilirubin'], ['bilirubin_direct', 'Bilirubin_direct'], ['calcium', 'Calcium ionized'], \
        ['chloride', 'Chloride'], ['creatinine', 'Creatinine'], \
        ['fibrinogen', 'Fibrinogen'], ['hematocrit', 'Hematocrit'], ['hemoglobin', 'Hemoglobin'], \
        ['so2', 'Oxygen saturation'], ['pco2', 'Partial pressure of carbon dioxide'], \
        ['platelet', 'Platelets'], ['potassium', 'Potassium'], ['inr', 'Prothrombin time INR'],
        ['pt', 'Prothrombin time PT'], \
        ['resp_rate', 'Respiratory rate'], ['sodium', 'Sodium'], ['wbc', 'White blood cell count'], \
        ['lactate', 'Lactic acid'], ['ph', 'pH'], ['bun', 'Blood urea nitrogen']]

    for names in names_list:
        original = total.loc[:, idx[names[0], ['mean', 'count']]].copy(deep=True)
        makeups = chart_lab.loc[:, idx[names[1], ['mean', 'count']]].copy(deep=True)
        filled = combine_cols(makeups, original)
        total.loc[:, idx[names[0], ['mean', 'count']]] = filled.loc[:, ['mean', 'count']].values
        chart_lab.drop(names[1], axis=1, level=0, inplace=True)

    # In eicu mbp contains both invasive and non-invasive, so combine them
    names_list = [['dbp', 'Diastolic blood pressure'], ['dbp_ni', 'Diastolic blood pressure'], \
                  ['mbp', 'Mean blood pressure'], ['mbp_ni', 'Mean blood pressure'], \
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
    select i.subject_id, i.hadm_id, v.stay_id, v.starttime, v.stoptime as endtime, v.antibiotic, v.route, i.icu_intime, i.icu_outtime 
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
    SELECT cr.stay_id, MIN(cr.charttime) as starttime, MAX(cr.charttime) as endtime, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
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
    platelats_trans = gcp2df(query)
    platelats_trans = compile_intervention(platelats_trans, 'platelats_trans')
    intervention = intervention.merge(
        platelats_trans[['subject_id', 'hadm_id', 'stay_id', 'hours_in', 'platelats_trans']],
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
    
        select
        coll.stay_id, coll.charttime as starttime, coll.endtime, coll.amount as colloid_bolus,  i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
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
    
        select crys.stay_id, crys.charttime as starttime, crys.endtime, crys.amount as crystalloid_bolus, i.subject_id, i.hadm_id, i.icu_intime, i.icu_outtime
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
    select c.subject_id, c.hadm_id, i.stay_id, c.myocardial_infarct, c.congestive_heart_failure, c.peripheral_vascular_disease, c.cerebrovascular_disease, c.dementia, c.chronic_pulmonary_disease, c.rheumatic_disease, c.peptic_ulcer_disease, c.mild_liver_disease, c.diabetes_without_cc, c.diabetes_with_cc, c.paraplegia, c.renal_disease, c.malignant_cancer, c.severe_liver_disease, c.metastatic_solid_tumor, c.aids
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
    return 


def extract_eicu(args):
    return
