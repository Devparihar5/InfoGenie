# InfoGenie:Research Assistant

## Description

The Research Assistant is a powerful research assistant powered by the Gemini Pro generative model. It streamlines the process of gathering information from web pages, breaking them down into smaller, digestible chunks, and embedding them into vectors for quick retrieval. When users pose questions, the Research Assistant leverages its capabilities to retrieve relevant context, generate prompts, and provide concise answers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Devparihar5/InfoGenie.git
   ```

2. Navigate to the project directory:
   ```bash
   cd InfoGenie
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

5. Install dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the Research Assistant, follow these steps:

1. Run the main script:
   ```bash
   streamlit run main.py
   ```

2. Input your query or question when prompted.

3. The Research Assistant will process your query, retrieve relevant information from web pages, generate a prompt, and provide you with a concise answer.

4. Continue to ask questions or end the session when you're done.

## Components

### Generative Model: Gemini Pro
Gemini Pro serves as the core of the Research Assistant, enabling it to understand and generate human-like text responses based on the provided context and user queries.

### Data Loading: URL Loader
The URL Loader component fetches web pages specified by users, allowing the Research Assistant to extract information from diverse online sources.

### Text Processing: Recursive Text Splitter
The Recursive Text Splitter breaks down the fetched web pages into smaller, more manageable chunks, facilitating efficient processing and analysis.

### Vector Embedding: GoogleGenerativeAIEmbeddings
GoogleGenerativeAIEmbeddings transforms the processed text chunks into numerical vectors, enabling the Research Assistant to perform similarity comparisons and retrieve relevant information quickly.

### Indexing: FAISS
FAISS (Facebook AI Similarity Search) is utilized for indexing the embedded vectors, enabling fast and efficient retrieval of relevant context and information.

### Pipeline: Langchain
The Langchain pipeline orchestrates the entire workflow of the Research Assistant, from loading and processing data to generating prompts and providing answers to user queries.

## Workflow

1. **Check User Query:** The Research Assistant examines the user's query to understand the information being sought.
  
2. **Load FAISS Index:** The indexed vectors containing information from fetched web pages are loaded into memory for quick access.

3. **Define Prompt Template:** A template for generating prompts based on the user query and context is defined.

4. **Retrieve Context:** Relevant context and information related to the user query are retrieved from the indexed vectors.

5. **Generate Prompt:** Using the retrieved context, the Research Assistant generates a concise prompt tailored to address the user's query.

6. **Provide Answer:** Finally, the Research Assistant provides the user with a well-informed answer based on the generated prompt.
