# Resume RAG Chat Assistant
An AI-powered chatbot that answers questions about a Keira's resume using Retrieval-Augmented Generation (RAG). The system loads PDF documents, creates embeddings, and uses OpenAI's GPT-3.5 to provide conversational responses in first-person.

Must Include:
1. Python 3.8+
2. OpenAI API Key

## Installation
1. Clone this repository
2. Pip install langchain openai gradio PyPDF2 python-dotenv
3. Create a .env file in the root directory and add your OpenAI API key: OPENAI_API_KEY=your_api_key_here

## Setup

1. **Prepare your documents:**
   - Place resume and LinkedIn PDFs in the designated directory
   - Update the path in Cell 4 (`load_pdfs` function) to your directory location

2. **Create summary file:**
   - Create a `me/summary.txt` file with a brief professional summary about yourself

3. **Update configuration:**
   - Modify the `resume_directory_path` in Cell 4 to point to your PDF directory
   - Optionally adjust the candidate's name in Cell 5 (Example: "Keira")

## Usage

### Running in Jupyter Notebook

Execute cells in order:

1. Cell 1-2: Import libraries and load environment variables
2. Cell 3-5: Initialize RAG components
3. Cell 6-7: Set up Gradio interface
4. Cell 8: Launch the chat interface

### Running as Python Script

Convert the notebook to a Python script or run:

```python
launch_chat()
```

The interface will launch and provide a local URL (and optionally a public share link).

## Customization

### Adjust Retrieval Parameters

In Cell 5, modify the `query` method:
- `k=3`: Number of document chunks to retrieve (default: 3)
- `chunk_size=1000`: Size of text chunks (default: 1000 characters)
- `chunk_overlap=200`: Overlap between chunks (default: 200 characters)

### Change LLM Model

In Cell 5, `chat` method, change:
```python
model="gpt-3.5-turbo"  # Can use gpt-4, gpt-4-turbo, etc.
```

### Adjust Response Length

In Cell 5, modify:
```python
max_tokens=400  # Increase for longer responses
```

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Ensure `.env` file exists in the root directory
- Verify the API key is correctly formatted

**"No documents found"**
- Check that PDFs are in the correct directory
- Verify filenames contain "KL" or "LINKEDIN"
- Ensure the path in `load_pdfs()` is correct

**Import errors**
- Install missing packages: `pip install [package-name]`
