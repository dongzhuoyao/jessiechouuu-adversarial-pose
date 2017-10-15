--- Load up network model or initialize from scratch
paths.dofile('models/' .. opt.netType .. '.lua')
paths.dofile('models/discriminator.lua')

-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/model_'..opt.lastEpoch..'.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
end

if not finalPredictions then
    if opt.netD ~= 'none' then
        assert(paths.filep(opt.netD), 'File not found: ' .. opt.netD)
        print('==> Loading model from: ' .. opt.netD)
        netD = torch.load(opt.netD)
    else
        netD = createModelD()
    end
end

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if not finalPredictions then
    criterionD_real = nn[opt.crit .. 'Criterion']()

    criterionG2 = nn.ParallelCriterion()
    criterionG2:add(nn[opt.crit .. 'Criterion'](), opt.lambda_G)
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
        
    if not finalPredictions then
        netD:cuda()
        criterionG2:cuda()
        criterionD_real:cuda()
    end
    
    cudnn.fastest = true
    cudnn.benchmark = true
end

k_t = opt.init_Kt
measure = 0.0
