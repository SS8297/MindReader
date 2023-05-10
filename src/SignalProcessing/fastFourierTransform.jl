####################################################################################################

"""

    extractFFT(edfDf::DataFrame, params::D)
    where D <: Dict

# Description
Use `extractFFT` on EDF file per channel from shell arguments. Returns a dictionary with channel names as keys.


See also: [`extractSignalBin`](@ref)
"""
function extractFFT(edfDf::DataFrame, params::D; window::Bool = false; onlyAbs::Bool = true) where D <: Dict
  if haskey(params, "window-size") && haskey(params, "bin-overlap")
    return extractFFT(edfDf, binSize = params["window-size"], binOverlap = params["bin-overlap"]; window = window, onlyAbs = onlyAbs)
  else
    @error "Variables are not defined in dictionary"
  end
end

####################################################################################################

"""

    extractFFT(channel::A;
    binSize::N, binOverlap::N)
    where A <: Array
    where N <: Number

# Description
Apply fast fourier transform (FFT) to channel.


See also: [`extractSignalBin`](@ref)
"""
function extractFFT(channel::A; binSize::N, binOverlap::N, window::Bool = false, onlyAbs::Bool = true) where A <: Array where N <: Number
  # define variables
  stepSize = floor(Int32, binSize / binOverlap)
  signalSteps = 1:stepSize:length(channel)
  freqAr = Array{Float64}(
    undef,
    onlyAbs ? 1 : 2,
    length(signalSteps),
    binSize
  )

  # iterate over signal bins
  for ι ∈ eachindex(signalSteps)
    signalBoundry = signalSteps[ι] + binSize - 1

    # extract channel
    if signalBoundry <= length(channel)
      channelExtract = [
        channel[signalSteps[ι]:signalBoundry];
        zeros(binSize)
      ]

    # adjust last bin
    elseif signalBoundry > length(channel)
      channelExtract = [
        channel[signalSteps[ι]:end];
        (signalBoundry - length(channel) |> abs |> zeros);
        zeros(binSize)
      ]
    end

    function hanning(window<:Array)
      points = collect(range(-0.5, 0.5, length(window)))
      hann = map(x -> cos(pi*x)^2, points)
      return window .* hann
    end

    if(window) channelExtract = hanning(channelExtract) end


    # calculate fourier transform
    fftChannel = fft(channelExtract)
    magniFft = abs.(fftChannel)
    freqAr[1, ι, :] = magniFft[1:binSize]
    if (!onlyAbs) phaseFft = angle.(fftChannel); freqAr[2, ι, :] = phaseFft[1:binSize] end

  end
  return freqAr
end

####################################################################################################

"""

    extractFFT(edfDf::DataFrame;
    binSize::N, binOverlap::N)
    where N <: Number

# Description
Use `extractFFT` on EDF file per channel. Returns a dictionary with channel names as keys.


See also: [`extractSignalBin`](@ref)
"""
function extractFFT(edfDf::DataFrame; binSize::N, binOverlap::N, window::Bool = false, onlyAbs::Bool = true) where N <: Number
  @info "Extracting channel frequencies..."
  channelDc = Dict()
  freqAr = begin
    stepSize = floor(Int32, binSize / binOverlap)
    signalSteps = 1:stepSize:size(edfDf, 1)
    Array{Float64}(
      undef,
      onlyAbs ? 1 : 2,
      binSize,
      length(signalSteps)
    )
  end

  # iterate on dataframe channels
  for (ψ, ε) ∈ enumerate(names(edfDf))
    α = extractFFT(edfDf[:, ψ], binSize = binSize, binOverlap = binOverlap, window = window, onlyAbs = onlyAbs)
    for β ∈ axes(α, 2)
      freqAr[:, :, β] = α[:, β, :]
    end
    channelDc[ε] = copy(freqAr)
  end
  return channelDc
end

####################################################################################################

"""

    extractDWT(edfDf::DataFrame, params::D)
    where D <: Dict

# Description
Use `extractDWT` on EDF file per channel from shell arguments. Returns a dictionary with channel names as keys.


See also: [`extractSignalBin`](@ref)
"""
function extractDWT(edfDf::DataFrame, params::D) where D <: Dict
  if haskey(params, "window-size") && haskey(params, "bin-overlap")
    return extractDWT(edfDf, binSize = params["window-size"], binOverlap = params["bin-overlap"])
  else
    @error "Variables are not defined in dictionary"
  end
end

####################################################################################################

"""

    extractDWT(channel::A;
    binSize::N, binOverlap::N)
    where A <: Array
    where N <: Number

# Description
Apply discrete wavelet transform (DWT) to channel.


See also: [`extractSignalBin`](@ref)
"""
function extractDWT(channel::A; binSize::N, binOverlap::N) where A <: Array where N <: Number
  # define variables
  stepSize = floor(Int32, binSize / binOverlap)
  signalSteps = 1:stepSize:length(channel)
  freqAr = Array{Float64}(
    undef,
    length(signalSteps),
    binSize
  )

  # iterate over signal bins
  for ι ∈ eachindex(signalSteps)
    signalBoundry = signalSteps[ι] + binSize - 1

    # extract channel
    if signalBoundry <= length(channel)
      channelExtract = [
        channel[signalSteps[ι]:signalBoundry];
      ]

    # adjust last bin
    elseif signalBoundry > length(channel)
      channelExtract = [
        channel[signalSteps[ι]:end];
        (signalBoundry - length(channel) |> abs |> zeros)
      ]
    end

    # calculate fourier transform
    # dwtChannel = map(x -> dwt(x, wavelet(WT.haar)), channelExtract)
    dwtChannel = dwt(channelExtract, wavelet(WT.haar))
    # Incase of complex wavelet, currently redundant
    false ? realDwt = abs.(dwtChannel) : realDwt = dwtChannel
    freqAr[ι, :] = realDwt

  end
  return freqAr
end

####################################################################################################

"""

    extractDWT(edfDf::DataFrame;
    binSize::N, binOverlap::N)
    where N <: Number

# Description
Use `extractDWT` on EDF file per channel. Returns a dictionary with channel names as keys.


See also: [`extractSignalBin`](@ref)
"""
function extractDWT(edfDf::DataFrame; binSize::N, binOverlap::N) where N <: Number
  @info "Extracting channel frequencies..."
  channelDc = Dict()
  freqAr = begin
    stepSize = floor(Int32, binSize / binOverlap)
    signalSteps = 1:stepSize:size(edfDf, 1)
    Array{Float64}(
      undef,
      1,
      binSize,
      length(signalSteps)
    )
  end

  # iterate on dataframe channels
  for (ψ, ε) ∈ enumerate(names(edfDf))
    α = extractDWT(edfDf[:, ψ], binSize = binSize, binOverlap = binOverlap)
    for β ∈ axes(α, 1)
      freqAr[:, :, β] = α[β, :]
    end
    channelDc[ε] = copy(freqAr)
  end
  return channelDc
end

####################################################################################################
"""

    extractCWT(edfDf::DataFrame, params::D)
    where D <: Dict

# Description
Use `extractCWT` on EDF file per channel from shell arguments. Returns a dictionary with channel names as keys.


See also: [`extractSignalBin`](@ref)
"""
function extractCWT(edfDf::DataFrame, params::D) where D <: Dict
  if haskey(params, "window-size") && haskey(params, "bin-overlap")
    return extractCWT(edfDf, binSize = params["window-size"], binOverlap = params["bin-overlap"])
  else
    @error "Variables are not defined in dictionary"
  end
end

####################################################################################################

"""

    extractCWT(channel::A;
    binSize::N, binOverlap::N)
    where A <: Array
    where N <: Number

# Description
Apply continuous wavelet transform (CWT) to channel.


See also: [`extractSignalBin`](@ref)
"""
function extractCWT(channel::A; binSize::N, binOverlap::N) where A <: Array where N <: Number
  # define variables
  stepSize = floor(Int32, binSize / binOverlap)
  signalSteps = 1:stepSize:length(channel)
  freqAr = Array{Float64}(
    undef,
    binSize÷4,
    binSize÷4,
    length(signalSteps)
  )

  # iterate over signal bins
  for ι ∈ eachindex(signalSteps)
    signalBoundry = signalSteps[ι] + binSize - 1

    # extract channel
    if signalBoundry <= length(channel)
      channelExtract = [
        channel[signalSteps[ι]:signalBoundry];
      ]

    # adjust last bin
    elseif signalBoundry > length(channel)
      channelExtract = [
        channel[signalSteps[ι]:end];
        (signalBoundry - length(channel) |> abs |> zeros)
      ]
    end

    # calculate fourier transfo
    # ContinuousWavelets.cwt(f, wavelet(cHaar, Q =85, β = 1))
    # cwtChannel = map( x -> ContinuousWavelets.cwt(x, wavelet(cHaar, Q = 85, β = 1)), channelExtract)
    #cwtChannel = ContinuousWavelets.cwt(channelExtract, wavelet(cHaar, Q = 85, β = 1))
    cwtChannel = ContinuousWavelets.cwt(channelExtract, wavelet(cHaar, Q = 66, β=4))[1:4:end,2:end]'
    # Incase of complex wavelet, currently redundant
    false ? realCwt = abs.(cwtChannel) : realCwt = cwtChannel
    freqAr[:, :, ι] = realCwt

  end
  return freqAr
end

####################################################################################################

"""

    extractCWT(edfDf::DataFrame;
    binSize::N, binOverlap::N)
    where N <: Number

# Description
Use `extractCWT` on EDF file per channel. Returns a dictionary with channel names as keys.


See also: [`extractSignalBin`](@ref)
"""
function extractCWT(edfDf::DataFrame; binSize::N, binOverlap::N) where N <: Number
  @info "Extracting channel frequencies..."
  channelDc = Dict()
  freqAr = begin
    stepSize = floor(Int32, binSize / binOverlap)
    signalSteps = 1:stepSize:size(edfDf, 1)
    Array{Float64}(
      undef,
      binSize÷4,
      binSize÷4,
      length(signalSteps)
    )
  end
 
  # iterate on dataframe channels
  for (ψ, ε) ∈ enumerate(names(edfDf))
    freqAr = extractCWT(edfDf[:, ψ], binSize = binSize, binOverlap = binOverlap)
    channelDc[ε] = copy(freqAr)
  end
  return channelDc 
end

####################################################################################################
