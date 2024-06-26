import base64
from io import BytesIO
import config
import logging
import asyncio
import time

import tiktoken
import openai
from volcenginesdkarkruntime import Ark
from logger import Logger



logger = Logger
# client = Ark(api_key="2ee3088d-bf0a-4510-8a9e-46784d29fbb4")
client = Ark(api_key=config.openai_api_key)

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "timeout": 60.0,
}


class Doubao:
    def __init__(self, model="Doubao-lite-4k"):
        assert model in {"Doubao-lite-4k"}, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"Doubao-lite-4k"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    logger.info(f"messages: {messages}, model: {self.model}")
                    r = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"Doubao-lite-4k"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    logger.info(f"messages: {messages}, model: {self.model}")
                    r_gen = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta

                        if "content" in delta:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = 0

                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                answer = self._postprocess_answer(answer)

            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    async def send_vision_message(
        self,
        message,
        dialog_messages=[],
        chat_mode="assistant",
        image_buffer: BytesIO = None,
    ):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model == "gpt-4-vision-preview" or self.model == "gpt-4o":
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, chat_mode, image_buffer
                    )
                    r = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message.content
                else:
                    raise ValueError(f"Unsupported model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = (
                    r.usage.prompt_tokens,
                    r.usage.completion_tokens,
                )
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError(
                        "Dialog messages is reduced to zero, but still has too many tokens to make completion"
                    ) from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(
            dialog_messages
        )

        return (
            answer,
            (n_input_tokens, n_output_tokens),
            n_first_dialog_messages_removed,
        )

    async def send_vision_message_stream(
        self,
        message,
        dialog_messages=[],
        chat_mode="assistant",
        image_buffer: BytesIO = None,
    ):
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model == "gpt-4-vision-preview" or self.model == "gpt-4o":
                    messages = self._generate_prompt_messages(
                        message, dialog_messages, chat_mode, image_buffer
                    )
                    
                    r_gen = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS,
                    )

                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            (
                                n_input_tokens,
                                n_output_tokens,
                            ) = self._count_tokens_from_messages(
                                messages, answer, model=self.model
                            )
                            n_first_dialog_messages_removed = (
                                n_dialog_messages_before - len(dialog_messages)
                            )
                            yield "not_finished", answer, (
                                n_input_tokens,
                                n_output_tokens,
                            ), n_first_dialog_messages_removed

                answer = self._postprocess_answer(answer)

            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e
                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (
            n_input_tokens,
            n_output_tokens,
        ), n_first_dialog_messages_removed

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _encode_image(self, image_buffer: BytesIO) -> bytes:
        return base64.b64encode(image_buffer.read()).decode("utf-8")

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode, image_buffer: BytesIO = None):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
                    
        if image_buffer is not None:
            messages.append(
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": message,
                        },
                        {
                            "type": "image_url",
                            "image_url" : {
                              
                                "url": f"data:image/jpeg;base64,{self._encode_image(image_buffer)}",
                                "detail":"high"
                            }
                        }
                    ]
                }
                
            )
        else:
            messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="Doubao-lite-4k"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "Doubao-lite-4k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            if isinstance(message["content"], list):
                for sub_message in message["content"]:
                    if "type" in sub_message:
                        if sub_message["type"] == "text":
                            n_input_tokens += len(encoding.encode(sub_message["text"]))
                        elif sub_message["type"] == "image_url":
                            pass
            else:
                if "type" in message:
                    if message["type"] == "text":
                        n_input_tokens += len(encoding.encode(message["text"]))
                    elif message["type"] == "image_url":
                        pass


        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):

        return 0, 0


async def is_content_acceptable(prompt):
    r = await openai.Moderation.acreate(input=prompt)
    return not all(r.results[0].categories.values())

# if __name__ == "__main__":
#     d = Doubao(model="Doubao-lite-4k")
#     try:
#         d.send_message_stream("hello", dialog_messages=[], chat_mode="cat")
#         time.sleep(1000000)
#     except Exception as e:
#         error_text = f"Something went wrong during completion. Reason: {e}"
#         logger.error(error_text)