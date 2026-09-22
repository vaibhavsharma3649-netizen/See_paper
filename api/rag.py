from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.load import load
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
load_dotenv()
title="Attention is all you need"
arxiv_id ='1706.03762'
pdfreader = PyMuPDFLoader(f'./papers/{arxiv_id}.pdf')
docs =pdfreader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunked = splitter.split_documents(docs)
from sentence_transformers import SentenceTransformer
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma(
    embedding_function=embedding,
    persist_directory=f'./vector_cache/{arxiv_id}',
    collection_name=f"paper_{arxiv_id.replace('.', '_')}"
)
vector_store.add_documents(chunked)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
query = "What is the research paper about ?"
retrieved_docs = retriever.invoke(query)
for i in retrieved_docs:
    print(i.page_content)
import json
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

from langchain_core.prompts import PromptTemplate
from langchain_core.load import dumpd,dumps

template = PromptTemplate(
        template='''
You are a Research Paper Visualization Agent.

Your job is to read the provided research paper and transform it into a simple, visual, beginner-friendly explanation.

Do NOT simply summarize the paper.
Your goal is to help the user understand:
1. What problem the paper solves
2. Why the problem matters
3. How the proposed method works
4. What happens to the data step-by-step
5. What the model/architecture looks like
6. What experiments were performed
7. What the results mean
8. What the limitations are
9. What the key takeaway is

Assume the user understands basic AI/ML concepts but may not understand the specific research topic.

-----------------------------------
PAPER
-----------------------------------

Title-{title}


Paper Summary-{context}

-----------------------------------
OUTPUT FORMAT
-----------------------------------

Return the explanation in the following structure:

# 1. Paper in One Sentence
Explain the entire paper in one very simple sentence.

# 2. The Problem
Explain:
- What problem is being solved?
- Why is it difficult?
- Why do existing approaches have limitations?

Use a simple real-world analogy when useful.

# 3. Core Idea
Explain the main idea of the paper in simple language.

Give the intuition before technical details.

# 4. How It Works

Create a step-by-step pipeline:

Input
 ↓
Step 1
 ↓
Step 2
 ↓
Step 3
 ↓
Output

For every step explain:
- What enters?
- What happens?
- What comes out?
- Why is this step necessary?

# 5. Architecture / Model

Convert the architecture into a clear conceptual diagram using text.

Example:

Input Image
     ↓
Feature Extractor
     ↓
Attention Layer
     ↓
Transformer
     ↓
Prediction

For each component explain its purpose in 1–3 sentences.

# 6. Important Concepts

List only the concepts necessary to understand this paper.

For each:

Concept:
Simple explanation:
Why it is used:

Avoid unnecessary mathematical jargon.

# 7. Mathematics

Only include equations that are essential for understanding the paper.

For every important equation:

Equation:
What each variable means:
Intuition:
What the equation is doing:

Do not include equations merely because they appear in the paper.

# 8. Training / Algorithm

Explain the training process as a sequence:

Data
 ↓
Preprocessing
 ↓
Model
 ↓
Loss
 ↓
Optimization
 ↓
Evaluation

Explain what happens at each stage.

# 9. Experiments

Explain:
- Dataset used
- Dataset size
- Baselines
- Metrics
- Experimental setup

Then explain why each experiment was performed.

# 10. Results

Present the important results in a simple table:

| Method | Metric | Result | What it means |
|--------|--------|--------|---------------|

Do not invent numbers.

If a result is not available in the paper, explicitly say:
"Not reported in the paper."

# 11. Why Does It Work?

Explain the reasoning behind the results.

Connect the architecture/method to the observed performance.

Clearly distinguish:
- What the authors demonstrated
- What is your interpretation

# 12. Limitations

Identify limitations explicitly mentioned by the authors.

Then separately identify reasonable limitations that can be inferred from the methodology.

Label inferred limitations as:
"Inferred limitation"

Do not present inferred limitations as claims made by the authors.

# 13. Compare With Previous Work

Explain:

Previous approach:
→ How it works

Proposed approach:
→ How it differs

Main difference:
→ Why the authors made the change

Do not claim that the proposed method is universally better.

# 14. Visual Mental Model

Finish with one simple diagram that captures the entire paper.

Example:

Problem
   ↓
Existing limitation
   ↓
Proposed idea
   ↓
Architecture
   ↓
Training
   ↓
Evaluation
   ↓
Results
   ↓
Conclusion

# 15. Explain Like I'm 15

Explain the entire paper using a simple real-world analogy.

Keep this section short.

# 16. Final Takeaway

Give exactly 5 bullet points containing the most important things the reader should remember.

-----------------------------------
IMPORTANT RULES
-----------------------------------

- Never invent information.
- Never invent experimental results, datasets, metrics, numbers, or claims.
- If information is missing, say "Not reported in the paper."
- Separate facts from your interpretation.
- Prefer intuition before mathematics.
- Use simple language.
- Define technical terms when first introduced.
- Do not explain every section of the paper if it is irrelevant to understanding the core idea.
- Focus on the contribution of the paper.
- Preserve the original meaning of the authors.
- Do not oversimplify technical details to the point of becoming incorrect.
- Use diagrams, arrows, tables, and structured layouts wherever they improve understanding.
- Keep paragraphs short.
- Highlight important terms using **bold text**.''',
input_variables=['title','context']
    )


prompt = template.invoke({
    'title': title,
    'context': context_text
})
model = ChatGroq(model='qwen/qwen3.8-27b')

result = model.invoke(prompt)
print(result.content)