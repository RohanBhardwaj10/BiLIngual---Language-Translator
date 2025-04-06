#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


# In[2]:


app = Flask(__name__)
model_path = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_path)
model = M2M100ForConditionalGeneration.from_pretrained(model_path)


# In[3]:


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


# In[4]:


def translate(text, src, tgt):
    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt))
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


# In[5]:


@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    if request.method == 'POST':
        src_lang = request.form['source_lang']
        tgt_lang = request.form['target_lang']
        text = request.form['text']
        src_lang_code = language_map[src_lang]
        tgt_lang_code = language_map[tgt_lang]
        translation = translate(text, src_lang_code, tgt_lang_code)
    return render_template('index.html', languages=language_map.keys(), translation=translation)


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




