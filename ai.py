from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import logging
import json
import os
import re
from collections import deque

import openai
from Bard import Chatbot as Bard
from EdgeGPT.EdgeGPT import Chatbot as Bing
from EdgeGPT.EdgeGPT import ConversationStyle
from hugchat import hugchat

MODELS = ["bard", "bing", "gpt", "hugchat", "llama"]


class ChatHistory:
    def __init__(self, msg_limit):
        self.stack = deque(maxlen=msg_limit)

    def append(self, msg):
        return self.stack.append(msg)

    def get_as_list(self):
        return list(self.stack)

    def get_as_string(self):
        res = ""
        for e in self.get_as_list():
            res +=  e + "\n"
        return res


class ChatModel:
    def __init__(self, model):
        assert (
            model in MODELS
        ), f"value attribute to {__class__.__name__} must be one of {MODELS}"
        self.model = model
        self.trigger = f"!{model}"
        self.api = self.get_api()

    def get_api(self):
        if self.model == "gpt":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_api_base = (
                os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
            )
            openai_model = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
            return OpenAIAPI(
                api_key=openai_api_key, api_base=openai_api_base, model=openai_model
            )

        if self.model == "llama":
            model = os.getenv("MODEL")
            return Llama_cpp(model=model)


class OpenAIAPI:
    def __init__(
        self, api_key, api_base, model="gpt-3.5-turbo", max_history=5, max_tokens=1024
    ):
        self.model = model
        self.history = ChatHistory(max_history)
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.api_base = api_base

    async def send(self, text):
        openai.api_key = self.api_key
        openai.api_base = self.api_base

        new_message = {"role": "user", "content": text}
        self.history.append(new_message)
        messages = self.history.get_as_list()

        response = openai.ChatCompletion.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens
        )

        self.history.append(response.choices[0].message)
        response = response.choices[0].message.content
        return response.strip()


class Llama_cpp:
    def __init__(self, model, max_history=5, max_tokens=1024):
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.model = model
        self.history = ChatHistory(max_history)
        self.max_tokens = max_tokens
        self.llm = LlamaCpp(
            model_path=model,
            temperature=1,
            #    model_pat
            max_tokens=1000,
            n_ctx=4096,
            top_p=1,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
            verbose=True, # Verbose is required to pass to the callback manager
        )
        self.prompt = PromptTemplate.from_template(
            """
            <|system|>: Vous êtes un assistant efficace. Vos réponses sont concises.
            <|user|>: {ctx}
            Question: {q}
            """
        )
        self.promptTxt = """
            <|system|>: Vous êtes l'assistant IA nommé Vigogne. Vos réponses sont concises.
            <|user|>: 
            """


        

    async def send(self, txt):
        h = self.history.get_as_string()
        message = self.prompt.format(ctx=h,q=txt)
        #message = self.promptTxt + txt
        #new_message = {"role": "<|user|>", "content": txt}
        self.history.append(txt)
        message_trace = "*********************** "+message
        logging.info(message_trace)
        response = ""
        try:
            response = self.llm(message)
            logging.info(message_trace+"\n"+response)
        except:
            return message  + "\n Quelque chose n'a pas bien fonctionné" 
        if response == "":
            return message + "\n Je n'ai rien à répondre à ça"
            
        return message + "\n" + response.strip()
    


