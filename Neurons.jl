
# Written by SKM 2015 
# Simulation of a network of Neurons in a heirarchial structure neurons < nets < layers
# 



## TO DO ## 
# 1. Function to allow more than two nets per layer
# 2.Build synapses both mean field and discrete events with plasticity

## Wish List ## 
# 1. Generalize to columnar struture
# 2. Instantiate Channels
# 3. Generalize, optimize, apply.


importall Neurocompute

import Base.convert
import Base.isless

# For storing variable V's time evolution, need to stipulate a conversion rule
convert(::Type{Float64}, x::Neuron) = x.V
convert(::Type{Float64}, x::IzNeuron) = x.V
convert(::Type{Float64}, x::EIzNeuron) = x.V

# For comparison with threshold values for V, defining isless operator
isless(x::Neuron, y::Float64) = (x.V<y)
isless(x::EIzNeuron, y::Float64) = (x.V<y)


timesteps = 1000
dt = 0.5
Enetsize = 800 # Excitatory net size . Try 1600E and 450I neurons
Inetsize = 200 # Inhibitor net size

Tnetsize = Enetsize + Inetsize

activity = Evolution(Tnetsize,timesteps) # Matrix to store the entire time trace of the variabe V
Enet = Net(Enetsize,EIzNeuron())         # Instantiating a net consisting of neurons of EIzNeuron type 
EnetConn = Conn(Enetsize)                # Defining Connection Matrix
Inet = Net(Inetsize,IIzNeuron())         # A net of size Inetsize and neuron type IIzNeuron
InetConn = Conn(Inetsize)                
Layer1 = Layer([Enet,Inet],2)            # Instantiating a Layer consisting of two nets

TnetConn = Conn(Enetsize,Inetsize)

using Bokeh

simsteps = 1                             # Number of Epochs each lasting timepoints * dt
timepoints = [1:timesteps]

for sim in 1:simsteps

    firings = Array(Int64,1,2)           # Declaring an empty array in Julia, creates a single element.
                                         # Careful! This first element needs to be removed later.!!
    for tstep in 1:timesteps



        #fired = find(x -> x >= 30., Enet.Neurons) # Can use this adhoc comparison since isless has been defined.
        #Ifired = find(x-> x >= 30., Inet.Neurons)
        
        fire!(Enet)                      # Firing the entire net. V's compared to thresholds, state=1 if V>threshold
        fire!(Inet)
        

        fired = find(x -> x.state == 1, Enet.Neurons)     # Collecting the indices of all neurons that fired
        Ifired = find(x -> x.state == 1, Inet.Neurons)
        
        
        Tfired = [fired; Enetsize .+ Ifired]              # Indices of all fired neurons in the whole layer 
                                                          # For layers with more than two nets, needs a function.

        firings = [firings; [tstep + 0*Tfired Tfired]]  

          

        #integrateN!(Enet,1,dt)             # Integrating each net individually!                
        #integrateN!(Enet,1,dt)
        #integrateN!(Inet,1,dt)
        #integrateN!(Inet,1,dt)

        integrateN!(Layer1,1,dt)           # Integrating the entire layer at once.
        integrateN!(Layer1,1,dt)           # Integrating twice for IzNeuron for numerical stability using dt=0.5

        activity.Block[1:Enetsize,tstep] = Enet.Neurons # Storing time evolution of Enet
        activity.Block[Enetsize+1:Tnetsize,tstep] = Inet.Neurons # Storing time evolution Inet
        
        #Updateinput(Enet,TnetConn,Tfired) # Updating current inputs for each net individually
        #Updateinput(Inet,TnetConn,Tfired)

        Updateinput(Layer1,TnetConn,Tfired) # Updating current inputs for entire layer.

    end

        firings = firings[2:end,:] # Starting index starts from 2 as empty matrix initialization has one element to start
        
        

        #plot(x,Fieldpotential')
        plot(firings[:,1],firings[:,2],"k.")
        hold(true)

        plot(timepoints,activity.Block[5,:]')
        showplot()
        hold(false)

end






