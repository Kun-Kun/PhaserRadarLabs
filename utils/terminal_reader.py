import array
import collections

import serial
import pydirectinput
import logging
import threading


class TerminalDataReader:

    def __init__(self, port, start_marker=65535, end_marker=65534):
        self.port = port
        self.startMarker = start_marker
        self.endMarker = end_marker
        self.data = collections.deque([], maxlen=10)
        self.lock = threading.Lock()

    def read_com_data(self):
        name = "data_reader"
        logging.info("Thread %s: starting", name)
        pydirectinput.FAILSAFE = False

        ser = serial.Serial(self.port, baudrate=115200, timeout=1)
        data_chunk = array.array("H")
        block = bytearray()
        while 1:
            block = block + ser.read_all()
            if len(block) > 1:
                if len(block) % 2 == 0:
                    data_chunk.frombytes(block)
                    del block[:]
                else:
                    data_chunk.frombytes(block[:-1])
                    block = block[-1:]
            if len(data_chunk) > 0:
                try:
                    start_pos = data_chunk.index(self.startMarker)
                    end_pos = data_chunk.index(self.endMarker)
                except ValueError:
                    continue
                self.lock.acquire()
                self.data.append(data_chunk[start_pos + 1:end_pos])
                self.lock.release()
                del data_chunk[start_pos:end_pos + 1]
        logging.info("Thread %s: finishing", name)

    def start(self):
        com_reader = threading.Thread(target=self.read_com_data, daemon=True)
        com_reader.start()

    def read(self):
        self.lock.acquire()
        if len(self.data) > 1:
            ret = self.data.pop()
        else:
            if len(self.data) == 1:
                ret = self.data[0]
            else:
                ret = []
        self.lock.release() 
        return ret
