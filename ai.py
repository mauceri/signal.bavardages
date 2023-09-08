from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain import PromptTemplate

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
            res += res + e['role'] + ": " + e['content'] + "\n"
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
        if self.model == "bing":
            cookie_path = "./config/bing.json"
            if not os.path.exists(cookie_path):
                cookie_path = None
            return BingAPI(
                conversation_style=ConversationStyle.creative, cookie_path=cookie_path
            )

        if self.model == "bard":
            token = os.getenv("BARD_TOKEN")
            return BardAPI(token)

        if self.model == "gpt":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_api_base = (
                os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
            )
            openai_model = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
            return OpenAIAPI(
                api_key=openai_api_key, api_base=openai_api_base, model=openai_model
            )

        if self.model == "hugchat":
            cookie_path = "./config/hugchat.json"
            return HugchatAPI(cookie_path=cookie_path)

        if self.model == "llama":
            model = os.getenv("MODEL")
            return Llama_cpp(model=model)


class BardAPI:
    def __init__(self, token):
        self.chat = Bard(token)

    async def send(self, text):
        return self.chat.ask(text)


class BingAPI:
    def __init__(self, conversation_style, cookie_path):
        self.conversation_style = conversation_style
        self.cookie_path = cookie_path
        self.cookies = self._parse_cookies()
        self.chat = Bing(cookies=self.cookies)

    def _parse_cookies(self):
        if self.cookie_path is None:
            return None
        else:
            return json.loads(open(self.cookie_path, encoding="utf-8").read())

    def _cleanup_footnote_marks(self, response):
        return re.sub(r"\[\^(\d+)\^\]", r"[\1]", response)

    def _parse_sources(self, sources_raw):
        name = "providerDisplayName"
        url = "seeMoreUrl"

        i, sources = 1, ""
        for source in sources_raw:
            if name in source.keys() and url in source.keys():
                sources += f"[{i}]: {source[name]}: {source[url]}\n"
                i += 1
            else:
                continue

        return sources

    async def send(self, text):
        data = await self.chat.ask(
            prompt=text, conversation_style=self.conversation_style
        )
        sources_raw = data["item"]["messages"][1]["sourceAttributions"]
        if sources_raw:
            sources = self._parse_sources(sources_raw)
        else:
            sources = ""

        response_raw = data["item"]["messages"][1]["text"]
        response = self._cleanup_footnote_marks(response_raw)

        if sources:
            return f"{response}\n\n{sources}"
        else:
            return response


class HugchatAPI:
    def __init__(self, cookie_path):
        self.chat = hugchat.ChatBot(cookie_path=cookie_path)

    async def send(self, text):
        return self.chat.chat(text)


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
        self.prompt_template = PromptTemplate.from_template("Répondez en français : {question}")
        logging.info("Dans le constructeur de Llama_cpp")
        self.llm = LlamaCpp(
            #    model_path="/Users/mauceric/PRG/llama.cpp/models/7B/ggml-model-q4_0.bin",
            model_path=model,
            temperature=1,
            max_tokens=500,
            top_p=1,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
            verbose=True, # Verbose is required to pass to the callback manager
        )
        prompt = """
        Vous parlez français et vous ne vous exprimez que dans cette langue. vous êtes concis dans vos réponses.
        """
        new_message = {"role": "user", "content": prompt}
        self.history.append(new_message)
        response = self.llm(prompt)
        new_message = {"role": "IA", "content": response}
        self.history.append(new_message)


        

    async def send(self, txt):
        new_message = {"role": "user", "content": txt}
        self.history.append(new_message)
        messages = self.history.get_as_string()
        
        response = self.llm(messages)

        new_message = {"role": "IA", "content": response.strip()}
        self.history.append(new_message)
        return response.strip()
