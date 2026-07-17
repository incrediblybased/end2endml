import sys
import logging

def error_message_details(error, error_details:sys):
    """This function will return the error message details"""
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    error_message = f'Error occurred in Python script name [{file_name}], line number [{lineno}], error message: [{error}]'
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details)
    
    def __str__(self):
        return self.error_message

