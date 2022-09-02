import argparse
from extract_database import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse to query MIMIC/eICU data")
    parser.add_argument("--database", type=str, default='MIMIC', choices=['MIMIC', 'eICU'])
    parser.add_argument("--age_min", type=int, default=18, help='Min patient age to query')
    parser.add_argument("--los_min", type=int, default=24, help='Min ICU LOS in hour')
    parser.add_argument("--los_max", type=int, default=240, help='Max ICU LOS in hour')
    parser.add_argument("--exit_point", type=str, default='All', choices=['All', 'Raw'],
                        help='Where to stop the pipeline')
    parser.add_argument("--patient_group", type=str, default='None', choices=['None', 'sepsis-3'],
                        help='Specific groups to extract')
    parser.add_argument("--project_id", type=str, default='lucid-inquiry-337016',
                        help = 'Specify the Bigquery billing project')
    parser.add_argument("--output_dir", type=str, default='/content/output')
    args = parser.parse_args()
    if args.database == 'MIMIC':
        extract_mimic(args)
    elif args.database == 'eICU':
        extract_eicu(args)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
