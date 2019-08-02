"""
File: logger.py

Project: UAV-II

Author: Dominik Slomma
"""

"""
Function: write_to_log

Logs a given message to the log file.

Paramerters:
    msg - Message to be logged as String
    err - Is it a error?
"""
def write_to_log(msg="", err=False):
    if err:
        msg = "Error: " + msg

    log_file = open('/tmp/UAV-II.log', 'a')
    log_file.write(msg)
    log_file.write('\n')
    log_file.close()
