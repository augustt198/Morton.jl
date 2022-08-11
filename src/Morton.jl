module Morton

export cartesian2morton, cartesian3morton
export morton2cartesian, morton3cartesian
export tree2morton, tree3morton
export morton2tree, morton3tree
export tree2cartesian, tree3cartesian
export cartesian2tree, cartesian3tree

const has_bmi2 = Base.BinaryPlatforms.CPUID.test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_bmi2)

# Access to 32/64 bit pdep/pext instrinsics
pdep_instr(x::UInt32, y::UInt32) = ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32, (UInt32, UInt32), x, y)
pdep_instr(x::UInt64, y::UInt64) = ccall("llvm.x86.bmi.pdep.64", llvmcall, UInt64, (UInt64, UInt64), x, y)
pext_instr(x::UInt32, y::UInt32) = ccall("llvm.x86.bmi.pext.32", llvmcall, UInt32, (UInt32, UInt32), x, y)
pext_instr(x::UInt64, y::UInt64) = ccall("llvm.x86.bmi.pext.64", llvmcall, UInt64, (UInt64, UInt64), x, y)

# Use 32 bit pdep/pext for smaller types
pdep_instr(x::T, y::T) where {T<:Union{UInt8, UInt16}} = pdep(UInt32(x), UInt32(y)) % T
pext_instr(x::T, y::T) where {T<:Union{UInt8, UInt16}} = pext(UInt32(x), UInt32(y)) % T

# 128 bit pdep/pext: https://www.talkchess.com/forum3/viewtopic.php?f=7&t=78804#p913803
function pdep_instr(x::UInt128, y::UInt128)
    lo = pdep_instr(x % UInt64, y % UInt64)
    hi = pdep_instr((x >> count_ones(y % UInt64)) % UInt64, (y >> 64) % UInt64)
    lo | (hi << 64)
end

function pext_instr(x::UInt128, y::UInt128)
    lo = pdep_instr(x % UInt64, y % UInt64)
    hi = pdep_instr((x >> 64) % UInt64, (y >> 64) % UInt64)
    lo | (hi << count_ones(y % UInt64))
end

isbitset(x, bit) = ((x >> bit) & 1) == 1
n_ones(T, n) = (T(1) << n) - T(1) # return the lower n bits set to 1

# Create masks needed to emulate a PDEP instruction pdep(?, y)
# in a divide-and-conquer way. These are also used for PEXT.
function pdep_emu_masks(y::T) where {T <: Unsigned}
    nset = count_ones(y)
    # Mask to keep only the lower nset bits
    startmask = n_ones(T, nset)

    bitwidth = 8*sizeof(T)
    bitpower = trailing_zeros(bitwidth) # i.e. T is 2^bitpower bits wide
    masks = map((bitpower-1):-1:0) do pow
        mask = T(0)
        count = 0 # as we go through the bits of y, keep track of how many 1s so far

        for bitindex in 0:(bitwidth-1)
            if isbitset(y, bitindex)
                diff = bitindex - count # the bit at bitindex should be shifted by this
                already = diff & ~n_ones(T, pow+1) # how much of the shift has been done already
                if isbitset(diff, pow) # if the `pow` shifting step contributes to the total shift
                                       # (reminiscent of exponentiation by squaring)
                    mask |= T(1) << (count + already)
                end
                count += 1
            end
        end

        mask
    end

    y, startmask, tuple(masks...)
end

# Apply the masks to emulate PDEP on x
function pdep_emu(x::Unsigned, maskset)
    y, start, masks = maskset
    x = x & start
    for (i, mask) in enumerate(masks)
        shift = 1 << (length(masks) - i)
        # shift the masked bits and fix the unmasked ones
        x = ((x & mask) << shift) | (x & ~mask)
    end
    return x
end

# Apply the masks to emulate PEXT on x
function pext_emu(x::Unsigned, maskset)
    y, start, masks = maskset
    for (i, mask) in enumerate(reverse(masks))
        shift = 1 << (i - 1)
        x = ((x & (mask << shift)) >> shift) | (x & ~mask)
    end
    return x & start
end

# Use native pdep/pext if available
pdep(x, maskset) = has_bmi2 ? pdep_instr(x, maskset[1]) : pdep_emu(x, maskset)
pext(x, maskset) = has_bmi2 ? pext_instr(x, maskset[1]) : pext_emu(x, maskset)

# Create integer of type T the i-th bit (0-indexed) is set
# if func(N, i) is true, where N is the bitwidth of N.
function bitmask(T, func)
    m = T(0)
    bitwidth = 8*sizeof(T)
    for i = 0:(bitwidth-1)
        func(bitwidth, i) && (m |= T(1) << i)
    end
    return m
end

const mask_generators = Dict(
    # generalization of a 0x55555555 mask (every other bit set)
    :every2 => (N, idx) -> mod(idx, 2) == 0,
    # generalization of a 0x09249249 mask (every third bit set)
    :every3 => (N, idx) -> (idx < (N÷3)*3) && mod(idx, 3) == 0
)
# define getmasks_<name>(::type) for each key `name` in mask_generators:
for (name, gen) in mask_generators
    for type in (:UInt8, :UInt16, :UInt32, :UInt64, :UInt128)
        constname  = Symbol(:Mmasks_, type, :_, name)
        methodname = Symbol(:getmasks_, name)

        @eval const $constname = pdep_emu_masks(bitmask($type, $gen))
        @eval $methodname(::$type) = $constname
    end
end

# Reverse of widen
narrow(::Type{UInt16})  = UInt8
narrow(::Type{UInt32})  = UInt16
narrow(::Type{UInt64})  = UInt32
narrow(::Type{UInt128}) = UInt64
narrow(x::T) where {T}  = convert(narrow(T), x)

### 2D

function cartesian2morton(x::T, y::T) where {T <: Unsigned}
    enc2(val) = pdep(val, getmasks_every2(val)) # encode function

    enc2(widen(x)) | enc2(widen(y)) << 1
end

cartesian2morton(x::T, y::T) where {T <: Signed} = cartesian2morton(unsigned(x), unsigned(y))
cartesian2morton(c::AbstractArray{<:Integer}) = cartesian2morton(c[1], c[2])

function morton2cartesian(m::T) where {T <: Unsigned}
    dec2(val) = pext(val, getmasks_every2(val)) # decode function

    dec2(m) % narrow(T), dec2(m >> 1) % narrow(T)
end

morton2cartesian(m::Signed) = morton2cartesian(unsigned(m))

### 3D

function cartesian3morton(x::T, y::T, z::T) where {T <: Unsigned}
    enc3(val) = pdep(val, getmasks_every3(val)) # encode function

    enc3(widen(x)) | enc3(widen(y)) << 1 | enc3(widen(z)) << 2
end

cartesian3morton(x::T, y::T, z::T) where {T <: Signed} = cartesian3morton(unsigned(x), unsigned(y), unsigned(z))
cartesian3morton(c::AbstractArray{<:Integer}) = cartesian3morton(c[1], c[2], c[3])

function morton3cartesian(m::T) where {T <: Unsigned}
    dec3(val) = pext(val, getmasks_every3(val)) # decode function

    dec3(m) % narrow(T), dec3(m >> 1) % narrow(T), dec3(m >> 2) % narrow(T)
end

morton3cartesian(m::Signed) = morton3cartesian(unsigned(m))

# Documentation:
"""
    cartesian2morton(c::Vector) -> m::Integer

Given a 2-D Cartesian coordinate, return the corresponding Morton number.

# Examples
```jldoctest
julia> cartesian2morton([5,2])
19
```
"""
cartesian2morton

"""
    cartesian3morton(c::AbstractVector) -> m::Integer

Given a 3-D Cartesian coordinate, return the corresponding Morton number.

# Examples
```jldoctest
julia> cartesian3morton([5,2,1])
67
```
"""
cartesian3morton

"""
    morton2cartesian(m::Integer) -> [x,y]

Given a Morton number, return the corresponding 2-D Cartesian coordinates.

# Examples
```jldoctest
julia> morton2cartesian(19)
2-element Array{Int64,1}:
 5
 2
```
"""
morton2cartesian

"""
    morton3cartesian(m::Integer) -> [x,y,z]

Given a Morton number, return the corresponding 3-D Cartesian coordinates.

# Examples
```jldoctest
julia> morton3cartesian(67)
3-element Array{Int64,1}:
 5
 2
 1
```
"""
morton3cartesian


function _treeNmorton(t::AbstractVector{T}, ndim::Integer) where T<:Integer
    n=m=0
    it=length(t)
    ndim2=2^ndim
    while it>0
      m += (t[it]-1)*ndim2^n
      n += 1
      it-=1
    end
    m+1
end

"""
    tree2morton(t::AbstractVector) -> m::Integer

Given a quadtree coordinate, return the corresponding Morton number.

# Examples
```jldoctest
julia> tree2morton([2,1,3])
19
```
"""
tree2morton(t::AbstractVector{T}) where T<:Integer = _treeNmorton(t,2)

"""
    tree3morton(t::AbstractVector) -> m::Integer

Given a octree coordinate, return the corresponding Morton number.

# Examples
```jldoctest
julia> tree3morton([2,1,3])
67
```
"""
tree3morton(t::AbstractVector{T}) where T<:Integer = _treeNmorton(t,3)


function _mortonNtree(m::T, ndim::Integer) where T<:Integer
    t=T[]
    ndim2=2^ndim
    while true
        d,r = [divrem(m-1,ndim2)...]+[1,1]
        pushfirst!(t,r)
        d==1 && break
        m=d
    end
    t
end

"""
    morton2tree(m::Integer) -> t::AbstractVector

Given a Morton number, return the corresponding quadtree coordinate.

# Examples
```jldoctest
julia> morton2tree(19)
3-element Array{Any,1}:
 2
 1
 3
```
"""
morton2tree(m::Integer) = _mortonNtree(m,2)

"""
    morton3tree(m::Integer) -> t::AbstractVector

Given a Morton number, return the corresponding octree coordinate.

# Examples
```jldoctest
julia> morton3tree(67)
3-element Array{Any,1}:
 2
 1
 3
```
"""
morton3tree(m::Integer) = _mortonNtree(m,3)


function _treeNcartesian(t::AbstractVector{T}, ndim::Integer) where T<:Integer
    c = [((t[1]-1)>>b)&1+1 for b in 0:ndim-1]
    if length(t)>1
        cn = _treeNcartesian(t[2:end], ndim)
        m = 2^(length(t)-1)
        return [m*(c[i]-1)+cn[i] for i in 1:ndim]
    else
        return c
    end
end

"""
   tree2cartesian(t::AbstractVector) -> c::AbstractVector

Given quadtree coordinate, return the corresponding 2-D Cartesian coordinate.

# Examples
```jldoctest
julia> tree2cartesian([2,1,3])
2-element Array{Int64,1}:
 5
 2
```
"""
tree2cartesian(t::AbstractVector{T}) where T<:Integer = _treeNcartesian(t, 2)

"""
   tree3cartesian(t::AbstractVector) -> c::AbstractVector

Given octree coordinate, return the corresponding 3-D Cartesian coordinate.

# Examples
```jldoctest
julia> tree3cartesian([2,1,3])
3-element Array{Int64,1}:
 5
 2
 1
```
"""
tree3cartesian(t::AbstractVector{T}) where T<:Integer = _treeNcartesian(t, 3)


function _cartesianNtree(c::AbstractVector{T}, half, ndim::Integer) where T<:Integer
    t = 1
    for d=1:ndim
        t += 2^(d-1)*(c[d]>half)
    end
    if half>1
        return [t, _cartesianNtree(map(x->(x-1)%half+1,c), half>>1, ndim)...]
    else
        return [t]
    end
end

"""
   cartesian2tree(c::AbstractVector) -> t::AbstractVector

Given a 2-D Cartesian coordinate, return the corresponding quadtree coordinate.

# Examples
```jldoctest
julia> cartesian2tree([5,2])
3-element Array{Int64,1}:
 2
 1
 3
```
"""
cartesian2tree(c::AbstractVector{T}) where T<:Integer =
      _cartesianNtree(c, max(2,nextpow(2, widen(maximum(c))))>>1, 2)

"""
   cartesian3tree(c::AbstractVector) -> t::AbstractVector

Given a 3-D Cartesian coordinate, return the corresponding octree coordinate.

# Examples
```jldoctest
julia> cartesian3tree([5,2,1])
3-element Array{Int64,1}:
 2
 1
 3
```
"""
cartesian3tree(c::AbstractVector{T}) where T<:Integer =
      _cartesianNtree(c, max(2,nextpow(2, widen(maximum(c))))>>1, 3)

end # module
