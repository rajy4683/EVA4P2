"""
 Copyright (c) 2020 Alan Yorinks All rights reserved.

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE
 Version 3 as published by the Free Software Foundation; either
 or (at your option) any later version.
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU AFFERO GENERAL PUBLIC LICENSE
 along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
"""
import sys
import time

from pymata4 import pymata4
import paho.mqtt.client as mqtt
from collections import OrderedDict

DIGITAL_PIN = 7  # arduino pin number
LED_COUNT = 50

### Lame hardcoded colors for each Gesture
allowed_gestures = ["FIVE", "FOUR", "THREE", "TWO", "ONE","FIST", "YEAH","SPIDERMAN", "ROCK", "OK","___"]

gesture_to_colormap = OrderedDict()
gesture_to_colormap["FIVE"]=[LED_COUNT,200,200,200]
gesture_to_colormap["FOUR"]=[LED_COUNT,200,100,0]
gesture_to_colormap["THREE"]=[LED_COUNT,0,100,200]
gesture_to_colormap["TWO"]=[LED_COUNT,204, 255, 204]
gesture_to_colormap["ONE"]=[LED_COUNT,204, 255, 204]
gesture_to_colormap["FIST"]=[LED_COUNT,255,0,0]
gesture_to_colormap["YEAH"]=[LED_COUNT,255,255,255]
gesture_to_colormap["SPIDERMAN"]=[LED_COUNT,153, 0, 51]
gesture_to_colormap["ROCK"]=[LED_COUNT,255, 255, 102]
gesture_to_colormap["OK"]=[LED_COUNT,255,255,255]
gesture_to_colormap["___"]=[LED_COUNT,0,0,0]



def blink(my_board, gesture_received):
    """
    This function will to toggle a digital pin.

    :param my_board: an PymataExpress instance
    :param pin: pin to be controlled
    """
    print("Invoked blink")
    # set the pin mode
    #my_board.set_pin_mode_digital_output(pin)
    #for (i=0;i<10;++i):
    #my_board.digital_write(pin, 1)
    if gesture_received not in allowed_gestures:
        print("Invalid Gesture {}".format(gesture_received))
        return
    selected_colormap = gesture_to_colormap[gesture_received]
    #data=list([LED_COUNT])
    #data.extend(gesture_to_colormap[gesture_received])
    #print(data)
    #data = led_count.extend(selected_colormap)
    print("Detected gesture {} setting color: {}".format(gesture_received, gesture_to_colormap[gesture_received]))
    my_board._send_sysex(0x67, gesture_to_colormap[gesture_received])
    print("Sending done")
    #my_board._send_sysex(PrivateConstants.PIN_STATE_QUERY, [pin])
    #print(my_board.query_reply_data.get(0x71))

    # toggle the pin 4 times and exit
    #for x in range(4):
    #    print('ON')
    #    my_board.digital_write(pin, 1)
    #    time.sleep(1)
    #    print('OFF')
    #    my_board.digital_write(pin, 0)
    #    time.sleep(1)

    #my_board.shutdown()
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("Hellos/+")

def on_message(client, userdata, msg):
    incoming_gesture = msg.payload.decode('ascii')
    print("Topic: {} message {}".format(msg.topic, incoming_gesture))
    blink(board, incoming_gesture)

### Initialize the board
board = pymata4.Pymata4(com_port='/dev/ttyS5')
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("127.0.0.1", 1883, 60)

try:
    client.loop_forever()
except:
    board.shutdown()
    sys.exit(0)

#from pymata4.private_constants import PrivateConstants

"""
Setup a pin for digital output and output a signal
and toggle the pin. Do this 4 times.
"""

# some globals
#try:
#    blink(board, DIGITAL_PIN)
#except KeyboardInterrupt:
#    board.shutdown()
#    sys.exit(0)
