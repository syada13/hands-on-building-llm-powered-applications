'''
Build a copilot agent that will help us generate reviews about YouTube videos, as well as post those reviews on our social media with a nice description and related picture.We need our agent to perform the following steps:

a. Search and transcribe a YouTube video based on our input.
b. Based on the transcription, generate a review with a length and style defined by the user query.
c. Generate an image related to the video and the review.
'''
from langchain.tools import YouTubeSearchTool
tool = YouTubeSearchTool()
result = tool.run("Avatar: The Way of Water,1")
#The tool returns the URL of the video. To watch it, you can add it to https://youtube.com domain.
print(result)

#Whisper is a transformer-based model introduced by OpenAI
 #1.It splits the input audio into 30-second chunks, converting them into spectrograms (visual representations of sound frequencies).
 #2. It then passes them to an encoder.
 #3. The encoder then produces a sequence of hidden states that capture the information in the audio.
 #4. A decoder then predicts the corresponding text caption, using special tokens to indicate the task (such as language identification, speech transcription, or speech translation) and the output language.The decoder can also generate timestamps for each word or phrase in the caption.

from pytube import YouTube
import openai

def get_youtube_video(yt_url):
  yt = YouTube("https://youtube.com"+yt_url,use_oauth=True,allow_oauth_cache=True)
  print(f"YouTube video to be downloaded-{yt}")
  vpath = yt.streams.filter(progressive=True,file_extension='mp4').order_by('resolution').desc().first().downloaded()
  print(f"Downloaded video {vpath}")
  return vpath

def transcribe_youtube_video(video_url):
  print(f"transcribing {video_url}")
  audio_file = open(video_url, 'rb')
  result = openai.Audio.transcribe('wisper-1',audio_file)
  audio_file.close()
  return (result['text'])


class CustomYTTranscribeTool(BaseTool):
    name = "CustomeYTTranscribe"
    description = "transcribe youtube videos associated with someone"

    def _summarize(self, url_csv:str) -> str:
        values_list = url_csv.split(",")
        url_set = set(values_list)
        datatype = type(url_set)
        print(f"[YTTRANSCIBE***], received type {datatype} = {url_set}")

        transcriptions = {}

        for vurl in url_set:
            vpath = get_youtube_video(vurl)

            transcription = transcribe_youtube_video(vpath)
            transcriptions[vurl]=transcription

            print(f"transcribed {vpath} into :\n {transcription}")

        with open("transcriptions.json", "w") as json_file:
            json.dump(transcriptions, json_file)
            
        return transcription
    
    def _run(self, query: str) -> str:
        """Use the tool."""
        return self._summarize(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("YTSS  does not yet support async")




import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools,initialize_agent

load_dotenv()
openai_api_key=os.environ['OPENAI_API_KEY']

llm = OpenAI(temperature=0)
model = ChatOpenAI()
tools = []
tolls.apend(YouTubeSearchTool())
tools.append(CustomYTTranscribeTool())
tools.append(load_tools(['dalle-image-generator'])[0])
agent = initialize_agent(tools, model, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent.run("search a video trailer of Avatar: the way of water. Return only 1 video. transcribe the youtube video and return a review of the trailer. Generate an image based on the video transcription")
print(result)

