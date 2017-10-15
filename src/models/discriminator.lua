paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModelD()

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = nnlib.SpatialConvolution(ref.nOutChannels+3,64,3,3,1,1,1,1)(inp)           -- 64
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local r4 = Residual(128,128)(r1)
    local r5 = Residual(128,opt.nFeats)(r4)
    
    local out = {}
    local inter = r5

    local hg = hourglass(4,opt.nFeats,inter)

    -- Residual layers at output resolution
    local ll = hg
    for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
    -- Linear layer to produce predictions
    ll = lin(opt.nFeats,opt.nFeats,ll)

    -- Predicted heatmaps
    local tmpOut = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(ll)
    table.insert(out,tmpOut)

    -- Final model
    local model = nn.gModule({inp}, out)

    return model

end
