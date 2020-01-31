
"""
Colllates Survival Data. Call it like this:
python3 collate_survival_data.py --root_path /Users/piyush/Documents/KIRC_DEAD_XML/
"""


import glob
import pandas as pd
import re
import argparse


parser = argparse.ArgumentParser("Argument Parser for Collate Survival Data")
parser.add_argument("--root_path", required=True, help="Root Path where XML files reside")


def collate_survival_data(root_path):
    files = glob.glob("{}/*/*.xml".format(root_path))

    df = pd.DataFrame(columns=["patient_barcode", "days_to_death"])
    for file in files:
        txt = open(file, 'r').read()
        days_to_death = int(re.search("\d+", re.search(">\d+<", re.search("days_to_death.*>\d+<", \
                        txt).group(0)).group(0)).group(0))
        patient_barcode = re.search(">.*<", re.search("bcr_patient_barcode.*>*<", txt).group(0)).group(0)[1:-1]
        df = df.append({"patient_barcode": patient_barcode, "days_to_death": days_to_death}, ignore_index=True)
    df.to_csv("KIRC_survival.csv", index=False)


def main():
    args = parser.parse_args()
    root_path = args.root_path
    collate_survival_data(root_path)

if __name__=="__main__":
    main()
