# Roxy developer workflow commands

# Git operations
roxy git status: user.roxy_git_status()
roxy git commit: user.roxy_git_commit()
roxy git push: user.roxy_git_push()
roxy git pull: user.roxy_git_pull()

# Claude Code integration
roxy start coding: user.roxy_start_coding()
roxy code review: user.roxy_code_review()

# Project operations
roxy new project <user.name>: user.roxy_new_project(name=user.name)
roxy open project <user.name>: user.roxy_open_project(name=user.name)

# Testing
roxy run tests: user.roxy_run_tests()
roxy test coverage: user.roxy_test_coverage()
