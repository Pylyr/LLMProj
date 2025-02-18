**Key Challenges & Implemented Solutions**

1️. Handling Redundant and Noisy Text in Rotowire  

*Problem*:  
Rotowire game reports contain excessive information unrelated to structured data.  
Players and teams may be referred to in different ways (e.g., "Lakers" vs. "Los Angeles Lakers").
Some necessary information about teams and players is missing in the text but can be found in external sources.

*Solution*:
-  Summarization → Extracts key facts from long reports.  
-  NER (Named Entity Recognition) → Identifies players, teams, and numerical values.  
-  Wikipedia API Lookup → Fills in missing background knowledge.  


2️. Improving WikiTableText with Sentence Ranking and Contrastive Filtering 

*Problem*:  
WikiTableText contains many sentences that do not contribute to table generation.  
Duplicate or similar sentences increase noise in the dataset. 

*Solution*:  
-   Ranking Sentences by Importance → Prioritizes relevant sentences.  
-  Contrastive Filtering → Removes redundant or near-duplicate sentences.  


3️. Adding New Datasets & Expanding Domain Coverage  

-   MIMIC-III (Medical Data)  
Extracts structured medical records from unstructured patient reports.  
Combines multiple tables (e.g., ADMISSIONS.csv, DIAGNOSES_ICD.csv) to generate comprehensive datasets.  
Produces text summaries of hospital admissions and structured patient statistics tables.  
-  Kaggle Financial Reports  
Converts company financial statements into structured tables.  
Extracts key metrics: revenue, expenses, net income, cash flow.  
Uses external data sources to supplement missing values.

---


**Launch instructions**

`git clone https://github.com/shirley-wu/text_to_table.git`

`cd /content/`

`gdown "https://drive.google.com/uc?id=1zTfDFCl1nf_giX7IniY5WbXi9tAuEHDn"`

`from google.colab import drive`
`drive.mount('/content/drive', force_remount=True)`

`tar -xzvf /content/drive/MyDrive/bart.base.tar.gz -C /content/`

`unzip /content/data_release.zip -d /content/text_to_table/`

`sudo apt-get update -y`
`sudo apt-get install -y python3.8 python3.8-distutils python3.8-dev`

`update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1`
`update-alternatives --config python3`

`sudo apt-get install python3-pip`
`python3 -m pip install --upgrade pip --user`

`pip install virtualenv`
`virtualenv -p python3.8 py38_env`

`pip install pip==23.2.1`

`pip install -r /content/text_to_table/requirements.txt`

`git clone --branch v0.10.2 https://github.com/facebookresearch/fairseq.git`

`cd /content/fairseq/`
`pip install --editable .`
`cd /content`

`mv /content/new_datasets/* /content/text_to_table/data_preprocessing/`
`mv /content/new_preprocess/* /content/text_to_table/data_preprocessing/`


