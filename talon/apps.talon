# Roxy application control commands

# Open applications
roxy open <user.app>: user.roxy_open_app(app=user.app)
roxy launch <user.app>: user.roxy_open_app(app=user.app)

# Close applications
roxy close <user.app>: user.roxy_close_app(app=user.app)
roxy quit <user.app>: user.roxy_close_app(app=user.app)

# Common apps
roxy open cursor: user.roxy_open_app(app="Cursor")
roxy open chrome: user.roxy_open_app(app="Google Chrome")
roxy open safari: user.roxy_open_app(app="Safari")
roxy open terminal: user.roxy_open_app(app="Terminal")
roxy open vscode: user.roxy_open_app(app="Visual Studio Code")
