# Resume RAG Chat Assistant
An AI-powered chatbot that answers questions about Keira's resume using Retrieval-Augmented Generation (RAG). The system loads PDF documents, creates embeddings using local sentence-transformers, and uses Google Gemini to provide conversational responses in first-person.

Must Include:
1. Python 3.8+
2. Google Gemini API Key

## Installation
1. Clone this repository
2. Pip install langchain google-genai gradio PyPDF2 python-dotenv sentence-transformers
3. Create a `.env` file in the root directory and add your Gemini API key: `GOOGLE_API_KEY=your_api_key_here`

## Setup

1. **Prepare your documents:**
   - Place resume and LinkedIn PDFs in the designated directory
   - Update the path in `load_pdfs` function to your directory location

2. **Create summary file:**
   - Create a `me/summary.txt` file with a brief professional summary about yourself

3. **Update configuration:**
   - Modify the `resume_directory_path` in `load_pdfs` to point to your PDF directory
   - Optionally adjust the candidate's name in the `chat` method (Example: "Keira")

## Usage

### Running as Python Script

```python
python app.py
```

The interface will launch and provide a local URL (and optionally a public share link).

### Deploying on Hugging Face Spaces

1. Push this repository to a Hugging Face Space
2. Add `GOOGLE_API_KEY` as a Secret in your Space settings
3. The app will launch automatically

## Customization

### Adjust Retrieval Parameters

In the `query` method, modify:
- `k=5`: Number of document chunks to retrieve (default: 5)
- `chunk_size=1000`: Size of text chunks (default: 1000 characters)
- `chunk_overlap=200`: Overlap between chunks (default: 200 characters)

### Change LLM Model

In the `chat` method, change:
```python
model="gemini-2.0-flash"  # Can use gemini-1.5-pro, gemini-2.5-pro, etc.
```

### Adjust Response Length

In the `chat` method, modify:
```python
max_output_tokens=1500  # Increase for longer responses
```

## Troubleshooting

**"GOOGLE_API_KEY not found"**
- Ensure `.env` file exists in the root directory
- Verify the API key is correctly formatted
- On Hugging Face Spaces, confirm the key is added under Settings → Secrets

**"No documents found"**
- Check that PDFs are in the correct directory
- Verify filenames contain "KL" or "LINKEDIN"
- Ensure the path in `load_pdfs()` is correct

**Import errors**
- Install missing packages: `pip install [package-name]`
