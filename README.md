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


2️. Improving WikiTableText with Sentence Ranking, Contrastive Filtering, and GNN-based Attribute Selection.

*Problem*:  
WikiTableText contains many sentences that do not contribute to table generation.  
Duplicate or similar sentences increase noise in the dataset. 
Include irrelevant attributes that do not align well with the text.

*Solution*:  
-   Ranking Sentences by Importance → Prioritizes relevant sentences.  
-  Contrastive Filtering → Removes redundant or near-duplicate sentences.
-  GNN-based Attribute Selection → Enhances table accuracy by identifying which attributes should be included in the final structured representation.

Example:

Input Table (Before Filtering)

[['title', '1978 federation cup (tennis)'], 
 ['subtitle', 'qualifying round'], 
 ['date', '19 august'], 
 ['winning team', 'philippines'], 
 ['score', '3–0'], 
 ['losing team', 'thailand']] 

Text:
 "Philippines won Thailand with 3–0 during 1978 Federation Cup."

[('subtitle', 'qualifying round'), 
 ('winning team', 'philippines'), 
 ('score', '3–0'), 
 ('losing team', 'thailand')]  
What Changed?  
* Removed "title" and "date" since they were not explicitly mentioned in the text.
* Kept "winning team", "score", and "losing team" because they were directly referenced.
* Maintained "subtitle" for structural context.

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




