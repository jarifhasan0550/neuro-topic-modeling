import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. Load the "Research Studies"
# mimics the task of reviewing and curating published studies
df = pd.read_csv('abstracts.csv')

# 2. Vectorization (Turning text into data for the AI)
vectorizer = TfidfVectorizer(stop_words='english')
data_vectorized = vectorizer.fit_transform(df['text'])

# 3. Apply Topic Modeling (LDA)
# addresses the requirement to 'extract summary information'
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(data_vectorized)

# 4. Output the Findings
print("Neuroinformatics Topic Modeling Complete.")
print("The AI has identified the primary research themes from the abstracts.")
