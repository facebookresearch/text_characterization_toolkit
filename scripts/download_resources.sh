#! /bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Folder for all resources
DATA_DIR="$(cd $(dirname -- "$0") && cd .. && pwd)/data"
mkdir $DATA_DIR

# Log output of this script into file - hard to debug if launched from Python
exec > $DATA_DIR/download_log.txt 2>&1

# Download resources for Python NLP libraries used by TCT
echo "DOWNLOADING SPACY / NLTK resources..."
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# The remainder of the script will put files into the directory called "data". This is empty by default and
# is included in gitignore - we have to first run this script to fill it up. The code is 
# configured to use this location by default and will error out if this script has not been run.
DATA_DIR="$(cd $(dirname -- "$0") && cd .. && pwd)/data"
mkdir $DATA_DIR
echo "DOWNLOADING RESOURCES TO $DATA_DIR"

# We will discard stuff from here after everything has been converted / unzipped / etc
TMP_DIR="download_resources_tmp"
rm -r $TMP_DIR
mkdir $TMP_DIR

# Where this file and some other python scripts are
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

cd $TMP_DIR

# MRC word database
wget https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/1054/allzip
unzip allzip
unzip 1054.zip
python $SCRIPTS_DIR/parse_mrc_dct.py mrc2.dct $DATA_DIR/mrc2.csv

# Word frequencies (from SpaCy):
wget https://github.com/explosion/spacy-lookups-data/raw/master/spacy_lookups_data/data/en_lexeme_prob.json -P $DATA_DIR

# Hyphenation
wget https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/hyph_en_US.dic -P $DATA_DIR

# Word prevalence 
PREVALENCE_FILE="WordprevalencesSupplementaryfilefirstsubmission"
wget http://crr.ugent.be/papers/$PREVALENCE_FILE.xlsx
python $SCRIPTS_DIR/convert_xlsx_resources.py  $PREVALENCE_FILE.xlsx $DATA_DIR/$PREVALENCE_FILE.csv  # need "pip install xlsx2csv" for this

# AOA
wget http://crr.ugent.be/papers/AoA_51715_words.zip
unzip AoA_51715_words.zip
python $SCRIPTS_DIR/convert_xlsx_resources.py AoA_51715_words.xlsx $DATA_DIR/AoA_51715_words.csv

# CONCRETENESS
wget https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/MediaObjects/13428_2013_403_MOESM1_ESM.xlsx
python $SCRIPTS_DIR/convert_xlsx_resources.py 13428_2013_403_MOESM1_ESM.xlsx $DATA_DIR/13428_2013_403_MOESM1_ESM.csv

cd ..
rm -r $TMP_DIR

echo "SUCCESSFULLY DOWNLOADED ALL RESOURCES!"

