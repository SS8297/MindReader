####################################################################################################

import Flux: kaiming_normal
import Random: seed!

# three-layered autoencoder
"""

    buildAutoencoder(inputLayer::I;
    nnParams)
    where I <: Integer

# Description
Build a three-layered autoencoder.

# Arguments
`inputLayer` number of neurons on input.

`nnParams` neural network hyperparameters.


See also: [`modelTrain!`](@ref)
"""
function buildAutoencoder(inputLayer::F; nnParams) where F <: Float
  @info("Building three-layered autoencoder...")
  args = nnParams()
  num_lay = args.l + 1
  model = Array{Dense}(undef, num_lay)

  pl = inputLayer
  for (i,n) in zip(1:num_lay, push!(args.λ, 1))
    nl = Int(inputLayer * n)
    seed!(31415);
    model[i] = Dense(pl, nl, args.σ; init = kaiming_normal)
    pl = nl
  end

  return Chain(model...)
end

####################################################################################################
