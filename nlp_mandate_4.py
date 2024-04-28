import google.generativeai as genai
import spacy
from flask import Flask, request, render_template

app = Flask(__name__)
GOOGLE_API_KEY = "AIzaSyBUL1rSE7x3xuIG1-s7oXr21SxbQIue1po"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")
nlp = spacy.load("en_core_web_sm")
prompt = """Please provide a brief analysis of this legal document. 
    Include (if relevant) key details such as the parties involved, the nature of the case,obligations, the issues discussed, and the final judgment.
    Summarize the document in a concise and informative way, highlighting the key legal aspects.
    Also perform NER of the legal provided.
    Do not include irrelevant information or conversational phrases in the summary."""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text", "")
        try:
            summary = summarize_text(text)
        except Exception as e:
            return render_template("index.html", original_text=text, summary=None, error=str(e))
        return render_template("index.html", original_text=text, summary=summary)
    return render_template("index.html")


def preprocess_text(text):

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    processed_sentences = []
    pos_tags = []
    dependencies = []

    for sent in doc.sents:
        processed_tokens = []
        for token in sent:
            # Collect POS tags and dependency relations
            pos_tags.append((token.text, token.pos_))
            dependencies.append((token.text, token.dep_, token.head.text))

            # Filter out stopwords and punctuation
            if not token.is_stop and not token.is_punct:
                processed_tokens.append(token.text.lower())

        processed_sentence = " ".join(processed_tokens)
        processed_sentences.append(processed_sentence)

    filtered_text = " ".join(processed_sentences)
    return filtered_text, entities, pos_tags, dependencies


def format_summary(text):
    # Split the text into paragraphs
    paragraphs = text.strip().split("\n\n")

    # Process each paragraph
    formatted_text = ""
    for paragraph in paragraphs:
        # Check if the paragraph is a heading (enclosed in **)
        if paragraph.startswith("**") and paragraph.endswith("**"):
            # Remove the ** and wrap the heading in <h3> tags
            heading_text = paragraph.strip("*")
            formatted_text += f"<h3>{heading_text}</h3>"
        else:
            # Wrap the paragraph in <p> tags
            formatted_text += f"<p>{paragraph}</p>"

    return formatted_text


def summarize_text(text):
    preprocessed_text, entities, pos_tags, dependencies = preprocess_text(text)
    response = model.generate_content(prompt + preprocessed_text)
    return format_summary(response.text)


if __name__ == "__main__":
    app.run(debug=True)
