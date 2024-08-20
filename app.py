import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import Any

from add_document import initialize_vectorstore
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import MomentoChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler

CHAT_UPDATE_INTERVAL_SEC = 1

load_dotenv()

# 로그
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)

# 봇 토큰을 사용하여 앱을 초기화
app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True,
)


class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        self.update_count = 0

    def on_llm_new_token(self, token: str, **kwargs):
        self.message += token

        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f"{self.message}\n\n작성 중..."
            )
            self.last_send_time = now
            self.update_count += 1

            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_context = "OpenAI API에서 생성되는 정보는 부정확하거나 부적절할 수 있으며, 우리의 견해를 나타내지 않습니다."

        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
            {"type": "divider"},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": message_context}],
            },
        ]

        app.client.chat_update(
            channel=self.channel, ts=self.ts, text=self.message, blacks=message_blocks
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# @app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub("<@.*>", "", event["text"])

    # 게시글의 키(=Momento 키): 첫 번째=event["ts"], 그 이후 = event["thread_ts"]
    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\n작성 중...", thread_ts=thread_ts)
    ts = result["ts"]

    history = MomentoChatMessageHistory.from_client_params(
        id_ts,
        os.environ["MOMENTO_CACHE"],
        timedelta(hours=int(os.environ["MOMENTO_TTL"])),
    )

    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever()

    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "위의 대화에서, 대화와 관련된 정보를 찾기 위한 검색 쿼리를 생성해 주세요.",
            ),
        ]
    )

    rephrase_llm = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
    )

    rephrase_chain = create_history_aware_retriever(
        rephrase_llm, retriever, rephrase_prompt
    )

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "아래의 문맥만을 고려하여 질문하세요.\n\n{context}"),
            (MessagesPlaceholder(variable_name="chat_history")),
            ("user", "{input}"),
        ]
    )

    qa_llm = ChatOpenAI(
        model_name=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True,
        callbacks=[callback],
    )

    qa_chain = qa_prompt | qa_llm | StrOutputParser()

    conversational_retrieval_chain = (
        RunnablePassthrough.assign(context=rephrase_chain | format_docs) | qa_chain
    )

    ai_message = conversational_retrieval_chain.invoke(
        {"input": message, "chat_history": history.messages}
    )

    history.add_user_message(message)
    history.add_ai_message(ai_message)


def just_ack(ack):
    ack()


app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


def handler(event, context):
    logger.info("handler called")
    header = event["headers"]
    logger.info(json.dumps(header))

    if "x-slack-retry-num" in header:
        logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
        return 200

    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
