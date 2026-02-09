# Roxy main activation commands

# Wake word and activation
hey roxy: user.roxy_wake()
hey rusty: user.roxy_wake()

# Stop listening
roxy stop: user.roxy_stop_listening()
roxy sleep: user.roxy_stop_listening()

# Text mode commands
roxy type <user.text>: user.roxy_process_text(text=user.text)
