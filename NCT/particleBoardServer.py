import requests
from pyparticleio.ParticleCloud import ParticleCloud
import pyparticleio2 as pp

particle = pp.Particle(access_token="3bdfa0e2739b955a2664c371a4a8cdba056645f6")
devices = particle.list_devices()
print(devices)
device = devices[0]
print("Device=",device)
var = particle.get_variable("e00fce68159fdddab349b9fe", "myDouble")
print("My double=", var['result'])

myint = particle.call_function("e00fce68159fdddab349b9fe","brew", "coffee")
print("MyInt=",myint)

# URL_CLOUD = "https://api.particle.io/v1/devices/e00fce68159fdddab349b9fe/myDouble?access_token=3bdfa0e2739b955a2664c371a4a8cdba056645f6"
#
# access_token = "3bdfa0e2739b955a2664c371a4a8cdba056645f6"
# # Create and initailize the ParticleCloud object as we did in the client code
# particle_cloud = ParticleCloud(username_or_access_token=access_token)
#
# # List devices
# all_devices = particle_cloud.devices
# for device in all_devices:
#     # particle_cloud.devices.RoverBoron()
#     print("Device: {0}".format(device))
#
# print(particle_cloud.devices['RoverBoron'].functions)
# print(particle_cloud.devices['RoverBoron'].variables)
# #print(particle_cloud.devices['RoverBoron'].myDouble)
# params = {"access_token": "3bdfa0e2739b955a2664c371a4a8cdba056645f6"}
# #res = requests.get(URL_CLOUD, params=params)
# res = requests.get(URL_CLOUD)
# print(res)
#
# # spark.devices['captain_hamster']
# #result = particle_cloud.devices['RoverBoron'].functions['brew'].brewCoffee("coffee")
# #print("Result= ", result)
#
# print("done")
#
# # Call function
# #particle_cloud.internet_button1.ledOn('led1')
#
# # Whatever signal we want to send (i.e. buy, sell, update, etc.), we can publish it with the publish() method
# # 'Argon2' is the name of the particle device that is publishing the signal. This will be changed to which ever device is plugged into our NUC
# particle_cloud.devices['Argon2'].publish("nn13buy")
#
