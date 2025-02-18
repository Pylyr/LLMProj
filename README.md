🔹 Key Challenges & Implemented Solutions

1️. Handling Redundant and Noisy Text in Rotowire  
Problem:

Rotowire game reports contain excessive information unrelated to structured data.  
Players and teams may be referred to in different ways (e.g., "Lakers" vs. "Los Angeles Lakers").
Some necessary information about teams and players is missing in the text but can be found in external sources.


Solution:
✅ Summarization → Extracts key facts from long reports.
✅ NER (Named Entity Recognition) → Identifies players, teams, and numerical values.
✅ Wikipedia API Lookup → Fills in missing background knowledge.


2️. Improving WikiTableText with Sentence Ranking and Contrastive Filtering
Problem:

WikiTableText contains many sentences that do not contribute to table generation.
Duplicate or similar sentences increase noise in the dataset.
Solution:
✅ Ranking Sentences by Importance → Prioritizes relevant sentences.
✅ Contrastive Filtering → Removes redundant or near-duplicate sentences.


3️⃣ Adding New Datasets & Expanding Domain Coverage
🔹 MIMIC-III (Medical Data)
Extracts structured medical records from unstructured patient reports.
Combines multiple tables (e.g., ADMISSIONS.csv, DIAGNOSES_ICD.csv) to generate comprehensive datasets.
Produces text summaries of hospital admissions and structured patient statistics tables.
🔹 Kaggle Financial Reports
Converts company financial statements into structured tables.
Extracts key metrics: revenue, expenses, net income, cash flow.
Uses external data sources to supplement missing values.
