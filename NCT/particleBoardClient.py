from pyparticleio2.ParticleCloud import ParticleCloud


# These are the method handlers that will handle each signal received
def buy(self):
    # TODO:Replace with actual buying code
    print('buy')


def sell(self):
    # TODO: Replace with actual selling code
    print('sell')

# Create a particle_cloud object, which will handle all particle events
# And configure the object with the access token correlated with our account
particle_cloud = ParticleCloud(username_or_access_token="3bdfa0e2739b955a2664c371a4a8cdba056645f6")

# Set up signal observers to call respective method handlers when buy and sell signals are received
# 'Argon1' is the name of the particle device that is subscribing to the signals. This will need to change depending on which device the client is receiving
particle_cloud.devices['Argon1'].cloud_subscribe("nn13buy", buy)
particle_cloud.devices['Argon1'].cloud_subscribe("nn13sell", sell)

particle_cloud.devices