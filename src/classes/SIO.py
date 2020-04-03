import socketio
import time
from src.classes.Enums import StateBase, IntentBase, IntentFollowUp, RecognizedEntities, EntityRequirements


def SIO(chatbot, address):
    SIO.chatbot = chatbot
    SIO.additional_info_conv_history = None
    SIO.latest_parsed_user_msg = None

    sio = socketio.Client()

    @sio.event
    def connect():
        print('Connected to server.')

    @sio.event
    def disconnect():
        print('Disconnected from server.')

    @sio.event
    def connect_error():
        print('Connection Error!')

    @sio.on('user connect')
    def on_user_connect(data):
        print('Detected user connecting.')
        SIO.chatbot.wipe_history()
        SIO.chatbot.set_new_user_id()
        opening_msg = SIO.chatbot.update_history_and_generate_opening_msg()
        sio.emit('bot message', opening_msg)

    @sio.on('user message')
    def on_user_message(raw_user_text):
        print(f'Detected user message: {raw_user_text}')
        reply = SIO.chatbot.get_reply(raw_user_text)
        update_state_if_processing()
        sio.emit('bot message', reply)

        if SIO.chatbot.exit_conversation():
            return

    def update_state_if_processing():
        if SIO.chatbot.policy.visualizer.task:
            SIO.chatbot.state = StateBase.processing
            sio.start_background_task(wait_for_task_completion, SIO.chatbot.policy.visualizer, SIO.chatbot.conversation_history.get_latest_base_intent())
            return

    def wait_for_task_completion(visualizer, latest_intent, interval=5):
        # Ensure this is kept parallel with the similarly named method in ChatBot.py
        while True:
            if visualizer.is_task_complete():
                break
            time.sleep(interval)

        reply = visualizer.get_reply(intent=latest_intent)
        sio.emit('bot message', reply)
        SIO.chatbot.policy.visualizer.task = None
        SIO.chatbot.state = StateBase.selecting_results
        return

    # Execute
    sio.connect(address)
    sio.wait()
    return
