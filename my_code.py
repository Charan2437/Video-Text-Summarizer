from flask import Flask, request, jsonify
import moviepy.editor
from transformers import pipeline
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video():
    vid = request.json['video_path']

    video = moviepy.editor.VideoFileClip(vid)
    aud = video.audio
    aud.write_audiofile("audio.mp3")

    whisper = pipeline('automatic-speech-recognition', model='openai/whisper-medium', device=-1)
    audio_text = whisper('audio.mp3')

    GOOGLE_API_KEY = 'AIzaSyD4zM2T4xx5IEnDjLf2z5H_Idh5zBsypgU'
    llm = GoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a text summarizer. You need to summarize text concisely and accurately. If the text is in any other language than English, please translate it to English first."),
            ("user", "Can you summarize {text} text for me?"),
        ]
    )

    out_parser = StrOutputParser()
    chain = prompt | llm | out_parser
    text_input = audio_text
    ai_message = chain.invoke({"text": text_input})

    return jsonify({'summary': ai_message})

if __name__ == '__main__':
    app.run(debug=True)