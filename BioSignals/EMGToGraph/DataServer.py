#####################################################################
# DataServer.py
# Read data from EMG device and save it to file
#####################################################################
import sys
import logging
import asyncio
import threading

from typing import Any, Union

from bless import (  # type: ignore
    BlessServer,
    BlessGATTCharacteristic,
    GATTCharacteristicProperties,
    GATTAttributePermissions,
)

import csv
import datetime as dt

maxRunTime = 3600 # Maximum time this script will run (to avoid lengthly datafile)

firstByte = False  # remember if a new integer (2 Bytes) will be send
highByte = 0  # remember highByte of integer 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)

# NOTE: Some systems require different synchronization methods.
trigger: Union[asyncio.Event, threading.Event]
if sys.platform in ["darwin", "win32"]:
    trigger = threading.Event()
else:
    trigger = asyncio.Event()

# For future acknowledge
def read_request(characteristic: BlessGATTCharacteristic, **kwargs) -> bytearray:
    logger.debug(f"Reading {characteristic.value}")
    return characteristic.value

# Act on received data byte
def write_request(characteristic: BlessGATTCharacteristic, value: Any, **kwargs):
    global firstByte, highByte # Declare global, needed for async
    value = int.from_bytes(value) # Convert byte to int
    logger.debug(f"Read {value}")
    # Every set of two bytes is preceded by zero
    if (value == 0):
        firstByte = True
        return # Skip, next time we will get the highByte
    # If firstByte is set, remember highByte, else add lowByte and write to file
    if (firstByte):
        logger.debug(f"highByte : {value}")
        highByte = value
        firstByte = False # highByte set, reset
    else:
        # Recieved lowByte, add them:
        logger.debug(f"Adding {highByte}*256 to {value}")
        val = highByte*256 + value
        # Add timestamp
        tim = dt.datetime.now().strftime('%H:%M:%S.%f')
        # Save value to file
        with open(r'EMGData.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([tim,val])
    trigger.set()


async def run(loop):
    trigger.clear()
    # Instantiate the server
    my_service_name = "NextComputer"
    server = BlessServer(name=my_service_name, loop=loop)
    server.read_request_func = read_request
    server.write_request_func = write_request

    # Add Service
    my_service_uuid = "A07498CA-AD5B-474E-940D-16F1FBE7E8CD"
    await server.add_new_service(my_service_uuid)

    # Add a Characteristic to the service
    my_char_uuid = "51FF12BB-3ED8-46E5-B4F9-D64E2FEC021B"
    char_flags = (
        GATTCharacteristicProperties.read
        | GATTCharacteristicProperties.write
        | GATTCharacteristicProperties.indicate
    )
    permissions = GATTAttributePermissions.readable | GATTAttributePermissions.writeable
    await server.add_new_characteristic(
        my_service_uuid, my_char_uuid, char_flags, None, permissions
    )

    logger.debug(server.get_characteristic(my_char_uuid))
    await server.start()
    logger.debug("Advertising")
    if trigger.__module__ == "threading":
        trigger.wait()
    else:
        await trigger.wait()

    await asyncio.sleep(2)
    logger.debug("Updating")
    server.get_characteristic(my_char_uuid)
    server.update_value(my_service_uuid, "51FF12BB-3ED8-46E5-B4F9-D64E2FEC021B")
    await asyncio.sleep(maxRunTime)
    await server.stop()


loop = asyncio.get_event_loop()
loop.run_until_complete(run(loop))
