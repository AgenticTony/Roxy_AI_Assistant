# Roxy system control commands

# Volume control
roxy volume up: user.roxy_volume_change(amount=10)
roxy volume down: user.roxy_volume_change(amount=-10)
roxy volume mute: user.roxy_volume_mute()
roxy volume <user.amount> percent: user.roxy_volume_set(amount=user.amount)

# Brightness control
roxy brightness up: user.roxy_brightness_change(amount=10)
roxy brightness down: user.roxy_brightness_change(amount=-10)

# Screen commands
roxy sleep: user.roxy_system_sleep()
roxy lock: user.roxy_system_lock()
roxy shutdown: user.roxy_system_shutdown()

# Search
roxy search for <user.query>: user.roxy_web_search(query=user.query)
roxy google <user.query>: user.roxy_web_search(query=user.query)

# File operations
roxy find files named <user.pattern>: user.roxy_find_files(pattern=user.pattern)
