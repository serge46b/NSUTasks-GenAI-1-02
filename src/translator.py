from transformers import pipeline
pipe_en2ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
pipe_ru2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")


def translate_en(en: str) -> str:
    return pipe_en2ru(en)[0]['translation_text']

def translate_ru(ru: str) -> str:
    return pipe_ru2en(ru)[0]['translation_text']


if __name__ == "__main__":
    test_text = "Hello, how are you"
    print(f"Переводим тестовую фразу '{test_text}' английского на русский...")
    translation_result = translate_en(test_text)
    print(f"Результат:\n\t{translation_result}")
    back_translation_result = translate_ru(translation_result)
    print(f"Результат обратного перевода: {back_translation_result}")


