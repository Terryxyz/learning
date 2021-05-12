import serial
import signal
import time

class Arduino_motor:
    def __init__(self):
        self.serialPort = "/dev/Arduino"
        self.baudRate = 57600
        self.ser = serial.Serial(self.serialPort, self.baudRate, timeout=0.5)
        self.shortest_thumb_length = 0.1

    def myHandler(signum, frame):
        pass
    
    def softtip(self):
        signal.signal(signal.SIGALRM, self.myHandler)
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        self.ser.write('2000'.encode('utf-8'))
        signal.setitimer(signal.ITIMER_REAL, 0)
        time.sleep(1)
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        self.ser.write('2090'.encode('utf-8'))
        signal.setitimer(signal.ITIMER_REAL, 0)
        time.sleep(0.1)

    def hardtip(self):
        signal.signal(signal.SIGALRM, self.myHandler)
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        self.ser.write('2180'.encode('utf-8'))
        signal.setitimer(signal.ITIMER_REAL, 0)
        time.sleep(1)
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        self.ser.write('2090'.encode('utf-8'))
        signal.setitimer(signal.ITIMER_REAL, 0)
        time.sleep(0.1)

    def set_thumb_length_int(self,length_int, wait_time = 0.2):
        signal.setitimer(signal.ITIMER_REAL, 0.001)
        self.ser.write(('1'+str(length_int)).encode('utf-8'))
        signal.setitimer(signal.ITIMER_REAL, 0)
        time.sleep(wait_time)  

    def set_thumb_length(self,length, wait_time = 0.2):
        length_int = (length-self.shortest_thumb_length)*180/0.028
        self.set_thumb_length_int(length_int, wait_time) 

if __name__ == '__main__':
    motor_control = Arduino_motor()
    motor_control.set_finger_length(0, wait_time = 1)





