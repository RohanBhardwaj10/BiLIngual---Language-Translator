from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Initialize FastAPI app
app = FastAPI()

# Load the tokenizer and model
model_path = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_path)
model = M2M100ForConditionalGeneration.from_pretrained(model_path)

# Language mapping
language_map = {
    "English": "en",
    "French": "fr",
    "Hindi": "hi",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh",
    "Arabic": "ar",
    "Japanese": "ja",
    "Russian": "ru",
    "Italian": "it",
    "Portuguese": "pt",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Turkish": "tr",
    "Korean": "ko",
    "Urdu": "ur",
    "Swahili": "sw",
    "Indonesian": "id",
    "Vietnamese": "vi",
    "Dutch": "nl",
    "Polish": "pl",
    "Greek": "el",
    "Czech": "cs",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Thai": "th",
    "Ukrainian": "uk",
    "Malay": "ms",
    "Hebrew": "he",
    "Persian": "fa",
    "Pashto": "ps",
    "Sinhala": "si",
    "Kannada": "kn",
    "Marathi": "mr",
    "Malayalam": "ml",
    "Nepali": "ne",
    "Kazakh": "kk",
    "Uzbek": "uz",
    "Azerbaijani": "az",
    "Serbian": "sr",
    "Croatian": "hr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Bulgarian": "bg",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Estonian": "et",
    "Armenian": "hy",
    "Georgian": "ka",
    "Amharic": "am",
    "Somali": "so",
    "Zulu": "zu",
    "Xhosa": "xh",
    "Igbo": "ig",
    "Yoruba": "yo",
    "Hausa": "ha"
}

# Define request body model for translation requests
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

# Translation logic
def translate(text: str, src: str, tgt: str) -> str:
    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt))
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# FastAPI endpoint for translation
@app.post("/translate/")
async def translate_text(translation_request: TranslationRequest):
    src_lang_code = language_map.get(translation_request.source_lang)
    tgt_lang_code = language_map.get(translation_request.target_lang)
    
    if not src_lang_code or not tgt_lang_code:
        raise HTTPException(status_code=400, detail="Invalid source or target language")

    translation = translate(translation_request.text, src_lang_code, tgt_lang_code)
    return {"translation": translation}

# A basic health check endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the multilingual translation API!"}
