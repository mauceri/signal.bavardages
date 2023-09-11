from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from collections import deque

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

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




# Make sure the model path is correct for your system!
llm = LlamaCpp(
#    model_path="/Users/mauceric/PRG/llama.cpp/models/7B/ggml-model-q4_0.bin",
    model_path="./models/gguf-model-q4_0.bin",
    temperature=1,
    max_tokens=500,
    top_p=1,
    n_ctx=2048,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)




prompt = ChatPromptTemplate.from_template(
    """
    <s>[INST] <<SYS>>
    Vous parlez français et vous ne vous exprimez que dans cette langue. vous êtes concis dans vos réponses.
    <</SYS>>
    {h} {q}
    [/INST]
    """
)
history = ChatHistory(5)
message = {"role":"humain","content":"Comment allez-vous ?"}
ret=llm(prompt.format(h="",q=message["content"]))
message = {"role":"IA","content":ret}
history.append(message)
print(prompt.format(h=history.get_as_string(),q="FIN"))
message = {"role":"humain","content":"Que pouvez-vous me dire du roi de France Henri IV ?"}
history.append(message)
ret=llm(prompt.format(h=history.get_as_string(),q=message["content"]))
message = {"role":"IA","content":ret}
history.append(message)
print(prompt.format(h=history.get_as_string(),q="FIN"))
message = {"role":"humain","content":"Que pouvez-vous me dire du renard et du corbeau ?"}
history.append(message)
ret=llm(prompt.format(h=history.get_as_string(),q=message["content"]))
message = {"role":"IA","content":ret}
history.append(message)
print(prompt.format(h=history.get_as_string(),q="FIN"))
