def single_review(text):
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer=tokenizer)

    if detect(text) == 'en':
        tokens = tokenizer.encode(text, return_tensors='pt')
        if len(tokens[0]) <= 512:
            return nlp(text)[0]['label'].replace('LABEL_0', 'NEGATIVE').replace('LABEL_1', 'NEUTRAL').replace('LABEL_2', 'POSITIVE')
    else:
        text = GoogleTranslator(source='auto', target='en').translate(text)
        tokens = tokenizer.encode(text, return_tensors='pt')
        if len(tokens[0]) <= 512:
            return nlp(text)[0]['label'].replace('LABEL_0', 'NEGATIVE').replace('LABEL_1', 'NEUTRAL').replace('LABEL_2', 'POSITIVE')