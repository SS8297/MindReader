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
function buildAutoencoder(inputLayer::I; nnParams) where I <: Integer
  @info("Building three-layered autoencoder...")
  args = nnParams()
  seed!(31415);
  return Chain(
    Dense(inputLayer, args.λ, args.σ; init = kaiming_normal),
    Dense(args.λ, inputLayer, args.σ; init = kaiming_normal),
  )
end

####################################################################################################
