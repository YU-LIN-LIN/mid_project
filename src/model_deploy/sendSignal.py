import numpy as np
import serial
import time

waitTime = 0.1

# generate the waveform table
signalLength = 42 * 2 + 32 * 2 + 24 * 2
#t = np.linspace(0, 2*np.pi, signalLength)
#signalTable = (np.sin(t) + 1.0) / 2.0

song0 = [261, 261, 392, 392, 440, 440, 392 ,349, 349, 330, 330, 294, 294, 261,392, 392, 349, 349, 330, 330, 294,392, 392, 349, 349, 330, 330, 294,261, 261, 392, 392, 440, 440, 392,349, 349, 330, 330, 294, 294, 261]

song1 = [
  261, 294, 330, 261, 261, 294, 330, 261,
  330, 349, 392, 330, 349, 392,
  392, 440, 392, 349, 330, 261, 
  392, 440, 392, 349, 330, 261,
  261, 196, 261, 261, 196, 261]

song2 = [
  392, 330, 330, 349, 294, 294,
  261, 293, 329, 349, 392, 392, 392,
  392, 330, 330, 349, 294, 294, 
  261, 330, 392, 392, 261]

noteLength0 = [
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2]

noteLength1 = [
  2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 4, 2, 2, 4,
  1, 1, 1, 1, 2, 2,
  1, 1, 1, 1, 2, 2,
  2, 2, 4, 2, 2, 4]

noteLength2 = [
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 4]

songs = song0 + song1 + song2
noteLengths = noteLength0 + noteLength1 + noteLength2

# output formatter
formatter = lambda x: "%d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
for data in songs:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
for data in noteLengths:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Signal sended")