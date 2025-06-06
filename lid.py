
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

def load_model():
    model_name = "l3cube-pune/hing-bert-lid"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

def classify_code_mixed_sentence(nlp_pipeline, sentence):
    results = nlp_pipeline(sentence)
    return [{"word": r["word"], "lang": r["entity_group"]} for r in results]

def run_on_file(file_path="data/input.csv"):
    df = pd.read_csv(file_path)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    nlp_pipeline = load_model()
    all_outputs = []

    for idx, row in df.iterrows():
        sentence = row["text"]
        prediction = classify_code_mixed_sentence(nlp_pipeline, sentence)
        all_outputs.append({"original_text": sentence, "prediction": prediction})

    for out in all_outputs:
        print(f"Sentence: {out['original_text']}")
        for token in out["prediction"]:
            print(f"  {token['word']}: {token['lang']}")
        print("-" * 40)

if __name__ == "__main__":
    run_on_file()
