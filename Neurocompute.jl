module Neurocompute

using Distributions

# Defining heirarchy  Neuron < Net < Layer
# Neuron can be E or I generally speaking.

abstract Neuron
abstract ENeuron <: Neuron
abstract INeuron <: Neuron
abstract IzNeuron <: Neuron


# Exported types 
# Evolution : time evolution matrix
# Conn      : Connection matrix
# Net       : A group of Neurons, generally only of E or I kind
# Layer     : A group of Nets

export Neuron, ENeuron, INeuron, IzNeuron
export Pyramidal, Inhibitory
export EIzNeuron, IIzNeuron
export Net, Layer
export Evolution, Conn

# Exported functions

# Integrate!    : General method for integrating evolution equations
# IntegrateN!   : Method for multiple steps, calls Integrate!
# Updateinput   : Updates input currents to a neuron due to incoming connections from other neurons, nets and layers!  
#                 Can take an entire Layer as an argument                  

export integrate!, integrateN!, fire!, Updateinput


# Defining the normal distribution

GD = Normal()

# Generic neuron types
type Pyramidal <: ENeuron
    V::Float64
    Vt::Float64
    Pyramidal(Vi=-65.,Vt=20.) = new(Vi + rand(),Vt)
end


type Inhibitory <: INeuron
    V::Float64
    Vt::Float64
    Inhibitory (Vi=-65.,Vt=20.) = new(Vi + rand(),Vt)
end


# Izhikevich Neurons, see Izhikevich 2003
type EIzNeuron <: IzNeuron
    V::Float64
    U::Float64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    I::Float64           # Total input current to be calculated at the time of simulation.
    Thal_Input::Float64  # Thalamic Input.
    state::Bool
    Flag_Integrate::Bool # Flag to allow U to be updated only on alternate calls to integrate for num.stability ! 
    
    function EIzNeuron()
        a = 0.02
        b = 0.2
        c = -65. + 15*rand()^2
        d = 8. - 6*rand()^2
        V = -65.
        U = b*V
        I = 0.
        Thal_Input = 5.0
        state = 0
        Flag_Integrate = 0
        
        new(V,U,a,b,c,d,I,Thal_Input,state,Flag_Integrate)
    end
    
end

type IIzNeuron <: IzNeuron
    V::Float64
    U::Float64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    I::Float64
    Thal_Input::Float64
    state::Bool
    Flag_Integrate::Bool
    
    function IIzNeuron()
        a = 0.02 + 0.08*rand()
        b = 0.25 - 0.05*rand()
        c = -65. 
        d = 2.
        V = -65.
        U = b*V
        I = 0.
        Thal_Input = 2.0
        state = 0
        Flag_Integrate = 0
        
        new(V,U,a,b,c,d,I,Thal_Input,state,Flag_Integrate)
    end
    
end


# Type Net, a group of neurons where number of neurons  = sizeNet 
type Net{T<:Neuron}
    Neurons::Vector{T}
    sizeNet::Int64
    Net(N,x) = new([T() for i in 1:N],N)
end

Net{T<:Neuron}(N::Int64,x::T) = Net{T}(N,x)


# Type Layer, a group of nets where number of nets = sizeLayer
type Layer{T<:Net}
    L::Vector{T}
    sizeLayer::Int64

    Layer(x,N) = new([x[i] for i in 1:N], N)
end

Layer{T<:Net}(x::Array{T,1},N::Int64) = Layer{T}(x,N)

# DataMatrix to hold temporal evolution a variable
type Evolution
    Block::Array{Float64,2} # Matrix to store activity

    function Evolution(rows::Integer,cols::Integer)
        new(zeros(rows,cols))
    end

end


# Connection matrix 
type Conn
    CMat::Array{Float64,2}
    
    function Conn(Esize::Integer,Isize::Integer)

        Tsize = Esize + Isize

        mat = [0.5*rand(Esize,Tsize); -1*rand(Isize,Tsize)]
        
        # Zeroing out diagonal elements to remove self connections
        for diag in 1:Tsize  
            
            mat[diag,diag] = 0
        
        end
        
        new(mat)
            
    end
end

Conn(Esize::Integer) = Conn(Esize,0)


function integrate!(x::Neuron, dt::Number)

    x.V += x.CI*dt + sqrt(dt)*rand(GD)

end


function integrate!(x::IzNeuron, dt::Number)
    V = x.V
    U = x.U
    a = x.a
    b = x.b
    I = x.I
    Flag = x.Flag_Integrate

    x.V += dt*(0.04*V^2 + 5*V + 140 - U + I)

    if Flag == 1
    	x.U += a*(b*V - U)
    	x.Flag_Integrate = 0
    else 
    	x.Flag_Integrate = 1
    end

    
end


function integrate!(x::Net, dt::Number)

    for Neuron in x.Neurons
        integrate!(Neuron, dt)
    end

end


function integrate!(x::Layer, dt::Number)

    for Net in x.L
        integrate!(Net, dt)
    end

end


function integrateN!(x::Net, steps::Integer, dt=Number)

    for i in 1:steps
        integrate!(x,dt)
    end
end


function integrateN!(x::Layer, steps::Integer, dt=Number)

    for i in 1:steps
        integrate!(x,dt)    
    end
end


function fire!(x::Neuron)
    if x.V >= x.V_thresh
        x.V = x.V_reset
    end
end


function fire!(x::IzNeuron)
    if x.V >= 30.
        x.V = x.c
        x.U = x.U + x.d
        x.state = 1
    else x.state = 0
    end
end


function fire!(x::Net)

    for Neuron in x.Neurons
        fire!(Neuron)
    end

end


function Updateinput(x::Net,C::Conn,f::Array{Int64,1}) # Calculating input from the network
    netsize = x.sizeNet

    for neuron in 1:netsize

    	Thal_Input = x.Neurons[neuron].Thal_Input

        x.Neurons[neuron].I = Thal_Input * rand(GD) + sum(C.CMat[f,neuron])
    end
    
end


function Updateinput(x::Layer,C::Conn,f::Array{Int64,1}) # Calculating input from the network
    
    numnets = x.sizeLayer

    for net in 1:numnets

    	netsize = x.L[net].sizeNet

	    for neuron in 1:netsize

	    	Thal_Input = x.L[net].Neurons[neuron].Thal_Input

	        x.L[net].Neurons[neuron].I = Thal_Input * rand(GD) + sum(C.CMat[f,neuron])
	    end

    end

    
end

end