#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from datetime import datetime
import os
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse

# from loguru import logger
from pipecat.services.llm_service import FunctionCallParams
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from fastapi.middleware.cors import CORSMiddleware
from src.mongodb import mongo_db
from src.schema import Appointment
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema


load_dotenv(override=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

# Mount the frontend at /
# app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/")
def root():
    return {"message": "Hello, World!"}


async def create_appointment(
    params: FunctionCallParams, name: str, date: str, time: str, phone: str, notes: str
):
    """Create an appointment.

    Args:
        name: The name of the customer. e.g. John Doe
        date: The date of the appointment. YYYY-MM-DD format. e.g. 2026-02-08
        time: The time of the appointment. 24-hour format. HH:MM format. e.g. 10:00
        phone: The phone number of the customer. USA format. e.g. +1234567890
        notes: The notes of the appointment. e.g. Please call me back.
    """
    appointment = Appointment(name=name, date=date, time=time, phone=phone, notes=notes)
    result = mongo_db.insert_appointment(appointment)
    await params.result_callback(result)


# Define a function using the standard schema
create_appointment_function = FunctionSchema(
    name="create_appointment",
    description="Create an appointment",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the customer. e.g. John Doe",
        },
        "date": {
            "type": "string",
            "description": "The date of the appointment. YYYY-MM-DD format. e.g. 2026-02-08",
        },
        "time": {
            "type": "string",
            "description": "The time of the appointment. 24-hour format. HH:MM format. e.g. 10:00",
        },
        "phone": {
            "type": "string",
            "description": "The phone number of the customer. USA format. e.g. +1234567890",
        },
        "notes": {
            "type": "string",
            "default": "",
            "description": "The notes of the appointment. e.g. Please call me back. (optional)",
        },
    },
    required=[
        "name",
        "date",
        "time",
        "phone",
        "notes",
    ],
)

# Create a tools schema with your functions
tools = ToolsSchema(standard_tools=[create_appointment_function])


async def run_example(webrtc_connection: SmallWebRTCConnection):
    # logger.info(f"Starting bot")

    # Create a transport using the WebRTC connection
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-2-delia-en",
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Get the time first to keep the f-string readable
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a friendly and concise Barber Shop assistant. "
                f"To create an appointment, ask for the user's name, date, time, and phone. "
                f"Notes are optional; if not provided, pass an empty string to the function. "
                f"Once you have the details, use 'create_appointment'. "
                f"Current date and time: {current_datetime}."
                f"After successfully creating the appointment, thank the user and say goodbye."
            ),
        },
    ]

    context = LLMContext(messages, tools=tools)

    llm.register_direct_function(
        create_appointment,
        cancel_on_interruption=False,  # Don't cancel on interruption
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


# @app.get("/", include_in_schema=False)
# async def root_redirect():
#     return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        # logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            # logger.info(
            #     f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}"
            # )
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_example, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


mongo_db.run()
# mongo_db.insert_appointment(
#     Appointment(
#         name="John Doe",
#         date="2026-02-08",
#         time="10:00",
#         phone="1234567890",
#         notes="This is a test appointment",
#     )
# )
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
#     parser.add_argument(
#         "--host", default="localhost", help="Host for HTTP server (default: localhost)"
#     )
#     parser.add_argument(
#         "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
#     )
#     args = parser.parse_args()

#     uvicorn.run(app, host=args.host, port=args.port)
