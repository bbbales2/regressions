import logging
import jsonrpcserver
import jsonrpcclient

from typing import TextIO

def read_json_rpc(file : TextIO):
	content_length = None
	content_type = None

	while True:
		line = file.readline().strip()
		if len(line) == 0:
			break
		cmd, string_value = line.split(":")

		if cmd == "Content-Length":
			content_length = int(string_value)
		elif cmd == "Content-Type":
			content_type = string_value
		else:
			raise Exception("Unrecognized command!")

	if content_length is None:
		raise Exception("Content length never set!")

	content = file.read(content_length)
	logging.debug(f"Reading in message: {content}")

	return content

def write_json_rpc(string_obj, file : TextIO):
	output_string = f"Content-Length:{len(string_obj)}\r\n\r\n{string_obj}"
	logging.debug(f"Writing out message: {repr(output_string)}")
	file.write(output_string)
	file.flush()
