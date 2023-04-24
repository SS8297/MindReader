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
  # @info("Building three-layered autoencoder...")
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

# convolutional autoencoder
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
  # @info("Building three-layered autoencoder...")
  args = nnParams()
  kw = args.KW
  kh = args.KH
  ci = args.CI
  co = args.CO
  ks = args.KS
  zp = args.ZP
  num_lay = args.l + 1
  encode = Array{Conv}(undef, length(kw))
  decode = Array{Conv}(undef, length(kw))
  for i in eachindex(kw)
    j = length(kw) - i + 1
    seed!(31415);
    encode[i] = Conv((kw[i], kh[i]), ci[i] => co[i], args.σ; init = kaiming_normal, stride = ks[i], pad = zp[i])
    seed!(31415);
    decode[j] = ConvTranspose((kw[j], kh[j]), ci[j] => co[j], args.σ; init = kaiming_normal, stride = ks[j], pad = zp[j])
  end
  return Chain(encode..., decode...)
end

