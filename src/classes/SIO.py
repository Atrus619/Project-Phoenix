import socketio
from src.classes.ConversationHistory import ConversationHistory
import time


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
        opening_msg = SIO.chatbot.update_history_and_generate_opening_msg()
        sio.emit('bot message', opening_msg)

    @sio.on('user message')
    def on_user_message(raw_text):
        print(f'Detected user message: {raw_text}')
        if SIO.chatbot.policy.visualizer.is_task_in_progress():
            sio.emit('bot message', 'I apologize, I am currently handling your previous request. Please ask me again after this completes.')
        elif not SIO.chatbot.policy.is_seeking_additional_info():
            SIO.latest_parsed_user_msg = SIO.chatbot.interpreter.parse_user_msg(raw_text=raw_text)
            reply = SIO.chatbot.update_history_and_generate_reply(parsed_user_msg=SIO.latest_parsed_user_msg)
            sio.emit('bot message', reply)

            if 'Please wait a moment while I collect the relevant information for you.' in reply:
                sio.start_background_task(wait_for_task_completion, SIO.chatbot.policy.visualizer)

            if SIO.chatbot.exit_conversation():
                return
            if SIO.chatbot.policy.is_seeking_additional_info():  # Newly seeking additional information
                SIO.additional_info_conv_history = ConversationHistory()
        else:  # is seeking additional info
            # Check if additional information was sought and continue asking until all information determined for request
            reply = SIO.chatbot.update_history_and_get_more_information(input_msg=raw_text, original_parsed_user_msg=SIO.latest_parsed_user_msg,
                                                                        additional_info_conv_history=SIO.additional_info_conv_history)
            sio.emit('bot message', reply)

    def wait_for_task_completion(visualizer, interval=5):
        while True:
            if visualizer.is_task_complete():
                break
            time.sleep(interval)

        reply = visualizer.get_reply()
        sio.emit('bot message', reply)
        return

    sio.connect(address)
    sio.wait()
    return

