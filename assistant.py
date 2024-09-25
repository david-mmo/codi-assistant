import asyncio
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero, elevenlabs, google
import os
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Tu nombre es Onyx. Tu interfaz con los usuarios será de voz.."
                    "Adopta la voz y actitud de David Goggins. Prioriza mensajes que empujen a la gente a enfrentar y superar sus miedos, y a nunca conformarse con menos de lo que pueden lograr. Muestra una fuerte mentalidad de guerrero"
                    "Evita usar signos de puntuación como '*' o emojis. Responde concretamente."
                    "No uses el signo de '*'"
                    "El usuario con el que hablas tiene las siguientes propiedades:"
                    "{'employee_name': 'Joel','employee_lastname': 'Alvarado','cobertura': '0.8','efectividad_compra': '0.7','desempeño': '0.5','drop_size': '50','descuentos': '2','hit_rate': '200'}"
                    "Y tiene los siguientes objetivos:"
                    "{'cobertura': '0.9','efectividad_compra': '0.6','desempeno': '0.8','drop_size': '30','descuentos': '5','hit_rate': '500'}"
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="onyx"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    elevenlabs_tts = elevenlabs.TTS(model_id="eleven_turbo_v2_5")
    google_tts = google.TTS(language="es-US", gender="male", voice_name="es-US-Polyglot-1", speaking_rate = 1.2, pitch=-10.0)


    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(model="nova-2", language="es"),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=google_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("¡Hola! ¿En qué puedo ayudarte?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame


if __name__ == "__main__":
    os.environ["LIVEKIT_URL"] = "wss://gepp-demo-mxc38o6k.livekit.cloud"
    os.environ["LIVEKIT_API_KEY"] = "APIZNuxajEUReYU"
    os.environ["LIVEKIT_API_SECRET"] = "GOPOZQpGi0MpqB7s8vYHB5dEqEBt77cPJfoa69BLmf1"
    os.environ["DEEPGRAM_API_KEY"] = "c7ea1764836e87e6c8afd341df50c6daf598ff96"
    os.environ["ELEVEN_API_KEY"] = "sk_3362f763a26d3e6eadd8a8e50227010e81dadd7685762ee4"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tts-credentials.json"
    print(os.environ.get("OPENAI_API_KEY"))
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
