from google.colab import auth
from google.cloud import bigquery
import os
auth.authenticate_user()

""" 
This query generates patientunitstayid for patients who satisfies the sepsis-3 criteria in the eICU database
Sepsis-3: SOFA >= 2 and suspicion of infection (SOI). So the following query queries SOFA score and find all the 
patientunitstayid that has SOFA >=2 and then find intersection with those under SOI. SOI is supposed to consider
both culture and antibiotics administration. However, the culture is very poorly populated in the eICU database 
so only antibiotics administration is considered.
"""

project_id='lucid-inquiry-337016'
os.environ["GOOGLE_CLOUD_PROJECT"]=project_id
client = bigquery.Client(project=project_id)
def gcp2df(sql, job_config=None):
    query = client.query(sql, job_config)
    results = query.result()
    return results.to_dataframe()

query = \
    """
    WITH 
    pafi as
    (
      select pa.patientunitstayid
      , pa.chartoffset
      -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
      -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
      -- in this case, the SOFA score is 3, *not* 4.
      , case when ve.patientunitstayid is null then pao2fio2ratio else null end pao2fio2ratio_novent
      , case when ve.patientunitstayid is not null then pao2fio2ratio else null end pao2fio2ratio_vent
    
      FROM lucid-inquiry-337016.eICU_derived.pao2fio2ratio pa
      left join lucid-inquiry-337016.eICU_derived.Vent ve
        on pa.patientunitstayid = ve.patientunitstayid
        and pa.chartoffset >= ve.priorventstartoffset
        and pa.chartoffset >=0
        and pa.chartoffset <= ve.priorventendoffset
        and ve.Invasive = 1
    )
    , vs AS
    (
        
      select co.patientunitstayid, co.hours_in
      , min(vs.nibp_mean) as meanbp_min
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join physionet-data.eicu_crd_derived.pivoted_vital vs
        on co.patientunitstayid = vs.patientunitstayid
        and co.hours_in = FLOOR(vs.chartoffset/60)
      group by co.patientunitstayid , co.hours_in
    )
    , gcs AS
    (
      select co.patientunitstayid, co.hours_in
      , min(gcs.gcs) as gcs_min
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join physionet-data.eicu_crd_derived.pivoted_gcs gcs
        on co.patientunitstayid = gcs.patientunitstayid
        and co.hours_in = FLOOR(gcs.chartoffset/60)
      group by co.patientunitstayid , co.hours_in
    )
    , bili AS
    (
      select co.patientunitstayid, co.hours_in
      , max(lab.bilirubin) as bilirubin_max
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join physionet-data.eicu_crd_derived.pivoted_lab lab
        on co.patientunitstayid = lab.patientunitstayid
        and co.hours_in = FLOOR(lab.chartoffset/60)
      group by co.patientunitstayid , co.hours_in
    )
    , cr AS
    (
      select co.patientunitstayid, co.hours_in
      , max(lab.creatinine) as creatinine_max
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join physionet-data.eicu_crd_derived.pivoted_lab lab
        on co.patientunitstayid = lab.patientunitstayid
        and co.hours_in = FLOOR(lab.chartoffset/60)
      group by co.patientunitstayid , co.hours_in
    )
    , plt AS
    (
      select co.patientunitstayid, co.hours_in
      , min(pl.labresult) as platelet_min
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join lucid-inquiry-337016.eICU_derived.platelets pl
        on co.patientunitstayid = pl.patientunitstayid
        and co.hours_in = FLOOR(pl.labresult/60)
      group by co.patientunitstayid , co.hours_in
    )
    , pf AS
    (
      select co.patientunitstayid, co.hours_in
      , min(pafi.pao2fio2ratio_novent) AS pao2fio2ratio_novent
      , min(pafi.pao2fio2ratio_vent) AS pao2fio2ratio_vent
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      -- bring in blood gases that occurred during this hour
      left join pafi
        on co.patientunitstayid = pafi.patientunitstayid
        and co.hours_in = FLOOR(pafi.chartoffset/60)
      group by co.patientunitstayid , co.hours_in
    )
    
    -- sum uo separately to prevent duplicating values
    -- collapse vasopressors into 1 row per hour
    -- also ensures only 1 row per chart time
    , uo as
    (
      select co.patientunitstayid, co.hours_in
      -- uo
      , min(uo.urineoutput_24hr) as uo_24hr
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join lucid-inquiry-337016.eICU_derived.uo24h uo
        on co.patientunitstayid = uo.patientunitstayid
        and co.hours_in = FLOOR(uo.chartoffset/60)
      group by co.patientunitstayid , co.hours_in
    )
    
    , vaso AS
    (
        SELECT 
            co.patientunitstayid
            , co.hours_in
            , max(vag.dobutamine) as dobutamine, max(vag.norepin_high) as norepin_high, max(vag.norepin_low) as norepin_low, 	
            max(vag.epin_high) as epin_high, max(vag.epin_low) as epin_low, max(vag.dopa_low) as dopa_low, 
            max(vag.dopa_mid) as dopa_mid, max(vag.dopa_high) as dopa_high, 
        from lucid-inquiry-337016.eICU_derived.icu_hourly co
        LEFT JOIN lucid-inquiry-337016.eICU_derived.vasoagent vag
            on co.patientunitstayid = vag.patientunitstayid
            and co.hours_in = FLOOR(vag.chartoffset/60)
        group by co.patientunitstayid , co.hours_in
    )
    , scorecomp as
    (
      select
          co.patientunitstayid
        , co.hours_in
        , pf.pao2fio2ratio_novent
        , pf.pao2fio2ratio_vent
        , vaso.dobutamine
        , vaso.norepin_high
        , vaso.norepin_low
        , vaso.epin_high 
        , vaso.epin_low
        , vaso.dopa_low
        , vaso.dopa_mid
        , vaso.dopa_high
        , vs.meanbp_min
        , gcs.gcs_min
        , uo.uo_24hr
        -- labs
        , bili.bilirubin_max
        , cr.creatinine_max
        , plt.platelet_min
      from lucid-inquiry-337016.eICU_derived.icu_hourly co
      left join vs
        on co.patientunitstayid = vs.patientunitstayid
        and co.hours_in = vs.hours_in
      left join gcs
        on co.patientunitstayid = gcs.patientunitstayid
        and co.hours_in = gcs.hours_in
      left join bili
        on co.patientunitstayid = bili.patientunitstayid
        and co.hours_in = bili.hours_in
      left join cr
        on co.patientunitstayid = cr.patientunitstayid
        and co.hours_in = cr.hours_in
      left join plt
        on co.patientunitstayid = plt.patientunitstayid
        and co.hours_in = plt.hours_in
      left join pf
        on co.patientunitstayid = pf.patientunitstayid
        and co.hours_in = pf.hours_in
      left join uo
        on co.patientunitstayid = uo.patientunitstayid
        and co.hours_in = uo.hours_in
      left join vaso
        on co.patientunitstayid = vaso.patientunitstayid
        and co.hours_in = vaso.hours_in
    )
    , scorecalc as
    (
      -- Calculate the final score
      -- note that if the underlying data is missing, the component is null
      -- eventually these are treated as 0 (normal), but knowing when data is missing is useful for debugging
      select scorecomp.*
      -- Respiration
      , case
          when pao2fio2ratio_vent   < 100 then 4
          when pao2fio2ratio_vent   < 200 then 3
          when pao2fio2ratio_novent < 300 then 2
          when pao2fio2ratio_vent   < 300 then 2
          when pao2fio2ratio_novent < 400 then 1
          when pao2fio2ratio_vent   < 400 then 1
          when coalesce(pao2fio2ratio_vent, pao2fio2ratio_novent) is null then null
          else 0
        end as respiration
    
      -- Coagulation
      , case
          when (platelet_min < 20 and platelet_min >0) then 4
          when platelet_min < 50  then 3
          when platelet_min < 100 then 2
          when platelet_min < 150 then 1
          when platelet_min is null then null
          else 0
        end as coagulation
    
      -- Liver
      , case
          -- Bilirubin checks in mg/dL
            when bilirubin_max >= 12.0 then 4
            when bilirubin_max >= 6.0  then 3
            when bilirubin_max >= 2.0  then 2
            when bilirubin_max >= 1.2  then 1
            when bilirubin_max is null then null
            else 0
          end as liver
    
      -- Cardiovascular
      , case
          when (norepin_high = 1 or epin_high = 1 or dopa_high = 1) then 4
          when (norepin_low = 1 or epin_low = 1 or dopa_mid = 1) then 3
          when (dopa_mid = 1 or dobutamine = 1)  then 2
          when meanbp_min < 70 then 1
          when meanbp_min is null then null
          else 0
        end as cardiovascular
    
      -- Neurological failure (GCS)
      , case
          when (gcs_min >= 13 and gcs_min <= 14) then 1
          when (gcs_min >= 10 and gcs_min <= 12) then 2
          when (gcs_min >=  6 and gcs_min <=  9) then 3
          when  gcs_min <   6 then 4
          when  gcs_min is null then null
          else 0
        end as cns
    
      -- Renal failure - high creatinine or low urine output
      , case
        when (creatinine_max >= 5.0) then 4
        when (uo_24hr < 200 and uo_24hr >0) then 4
        when (creatinine_max >= 3.5 and creatinine_max < 5.0) then 3
        when (uo_24hr < 500 and uo_24hr >0) then 3
        when (creatinine_max >= 2.0 and creatinine_max < 3.5) then 2
        when (creatinine_max >= 1.2 and creatinine_max < 2.0) then 1
        when coalesce (uo_24hr, creatinine_max) is null then null
        else 0 
      end as renal
      from scorecomp
    )
    , score_final as
    (
      select s.*
        -- Combine all the scores to get SOFA
        -- Impute 0 if the score is missing
       -- the window function takes the max over the last 24 hours
        , coalesce(
            MAX(respiration) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0) as respiration_24hours
         , coalesce(
             MAX(coagulation) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
             ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
            ,0) as coagulation_24hours
        , coalesce(
            MAX(liver) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0) as liver_24hours
        , coalesce(
            MAX(cardiovascular) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0) as cardiovascular_24hours
        , coalesce(
            MAX(cns) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0) as cns_24hours
        , coalesce(
            MAX(renal) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0) as renal_24hours
    
        -- sum together data for final SOFA
        , coalesce(
            MAX(respiration) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0)
         + coalesce(
             MAX(coagulation) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
             ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0)
         + coalesce(
            MAX(liver) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0)
         + coalesce(
            MAX(cardiovascular) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0)
         + coalesce(
            MAX(cns) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0)
         + coalesce(
            MAX(renal) OVER (PARTITION BY patientunitstayid ORDER BY hours_in
            ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
          ,0)
        as sofa_24hours
      from scorecalc s
      WINDOW W as
      (
        PARTITION BY patientunitstayid
        ORDER BY hours_in
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING
      )
    ),
    sofa as 
    (
    select patientunitstayid, max(sofa_24hours) as sofa_highest
    from score_final
    GROUP BY patientunitstayid
    )
    
    SELECT patientunitstayid 
    FROM sofa
    WHERE sofa_highest >=2
    
    INTERSECT DISTINCT
    
    SELECT DISTINCT md.patientunitstayid
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
    AND FLOOR(LEAST(md.drugstopoffset, i.unitdischargeoffset)/60) > FLOOR(GREATEST(md.drugstartoffset, 0)/60)
    AND md.drugordercancelled = 'No'
    AND md.drugstartoffset is not null 
    AND md.drugstopoffset is not null
    """
patient = gcp2df(query)
