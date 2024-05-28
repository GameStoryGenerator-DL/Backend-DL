from flask import Flask, request, jsonify
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline

app = Flask(__name__)

CHROMA_PATH = "chroma"
# PROMPT_TEMPLATE = """
# Répondre à la question : "## {question} ##" en se basant sur le contexte suivant :

# {context}

# ---

# Répondez à cette {question} en se basant sur le contexte donné ci-dessus
# """

PROMPT_TEMPLATE = """
Générer une histoire de jeux vidéo en se basant sur cette phrase : "## {question} ##" et selon le contexte suivant :

{context}

---

Générer une histoire de jeux vidéo en se basant sur cette phrase : "## {question} ##" et selon le contexte donné ci-dessus
"""




@app.route('/', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')

    if not query_text:
        return jsonify({'error': 'Query text is required'}), 400

    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=1)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    generator = pipeline('text-generation', model='DLProjectLSI/Phi-3-mini-4k_16bit')
    # generator = pipeline('text-generation', model='DLProjectLSI/GPT2-MINI')
    response = generator(prompt, max_length=1024, num_return_sequences=1, truncation=True)
    response_text = response[0]['generated_text']

    # Remove the "Human" and prompt template part from the response
    if "et selon le contexte donné ci-dessus" in response_text:
        response_text = response_text.split("et selon le contexte donné ci-dessus", 1)[-1]
    
    formatted_response = {
        'response': response_text,
    }

    return response_text
    # return jsonify(formatted_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)