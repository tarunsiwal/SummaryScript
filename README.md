"# SummaryScript" 
This is a programe code that reads an article, processes it to remove stopwords, computes similarity between sentences, ranks the sentences using PageRank, and finally selects the top sentences to create a summary. This process involves several steps of text processing, vectorization, similarity calculation, and graph-based ranking.


Create vertual invironment using this command 
'python -m venv myenv'

Then activate it 
'myenv\Scripts\activate'

These are the dependencies
click==8.1.7
colorama==0.4.6
joblib==1.4.0
networkx==3.3
nltk==3.8.1
numpy==1.26.4
regex==2023.12.25
scipy==1.13.0
tqdm==4.66.2

How to add main python file to the vertual environment just use 
'python main.py'
