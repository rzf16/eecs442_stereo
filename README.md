# EECS 442 Final Project: Real-Time Stereo Vision
We implement a real-time stereo algorithm by [Ben-Tzvi and Xu](http://rmlab.org/pdf/C16_ROSE_2010.pdf). Images are captured using Raspberry Pi Cameras (PiCams) and sent to a laptop, where the stereo algorithm is run and the depth map is displayed.


## Building the RPI Zero image
You will need to setup ethernet over usb to work with a RPI zero. It should be pretty simple. 

- Flash an SD card using the official raspberrypi imager software. 
- Open up the root directory and 
- Paste into cmdline.txt: modules-load=dwc2,g_ether
- paste into config.txt: dtoverlay=dwc2
- Create an ***empty*** file in the root called ssh with ***no extension*** 
- Boot up your RPI
- Be confused why it isn't connecting
- On your Ubuntu 18.04 device do the following
- Go to settings, network, and click the cog for the card that says usb ethernet
- If you couldn't find the card watch that page and plug and unplug your RPI until it shows up
- Select ipv4 settings
- Change to local link only

to test that the connection worked open a terminal and type

    ping raspberrypi.local

If that works you should now be able to ssh into your rpi at `pi@raspberrypi.local` however you won't have access to the internet yet so now we need to set that up

On your Ubuntu machine run

    sudo apt-get update &&
    sudo apt-get install bridge-utils

Now you should be able to run 

    nm-connection-editor

Disconnect the RPI

This friendly GUI will help us setup a network profile for our new ethernet device. 

Select the automatically created profile for the pi, it should be in the ethernet section. Mine were normally named wired connection X where X is a number like 1, 2, 3. In the `Ethernet` tab take note of what the `device` selected was (ignore the MAC addr in the parens). Then delete that profile.

 Now create a new profile. In the `Ethernet` tab set the name to something that you can remember. Set the device to be the device from the previous profile ***without the stuff in parens***. My device was `enp0s20f0u3`. Then in the `IPv4` settings change the method to `Shared to other computers`. Save your changes.

 Plug in your RPI and if it connects right away to the profile you made that is a really good sign. Test pinging the pi again at raspberrypi.local, ssh into the pi and then try pinging 8.8.8.8. 

 If you can run

    ping 8.8.8.8

from within the RPI's ssh then you are all setup, otherwise go back and try recreating your network profile the error is going to be on your local machine's side.