
-- Prepare tensors for saving network output
local validSamples = opt.validIters * opt.validBatch
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end

-- Main processing step
function step(tag)
    local avgAcc = 0.0
    local avgG1, avgG2, avgD_real, avgD_fake = 0.0, 0.0, 0.0, 0.0
    local output, err, idx
    local output, err_G1, err_G2, output_real, output_fake, errD_real, errD_fake, errD, errG
    local df_G1, df_G2_out, df_G2_in, df_D_real, df_D_fake
    local param, gradparam = model:getParameters()
    local paramD, gradparamD = netD:getParameters()
    local function evalFn(x) return errG, gradparam end
    local function evalFnD(x) return errD, gradparamD end

    if tag == 'train' then
        model:training()
        netD:training()
        set = 'train'
    else
        model:evaluate()
        if tag == 'predict' then
            print("==> Generating predictions...")
            local nSamples = dataset:size('test')
            saved = {idxs = torch.Tensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim))}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            set = 'test'
        else
            set = 'valid'
        end
    end    

    local nIters = opt[set .. 'Iters']
    for i,sample in loader[set]:run() do
        xlua.progress(i, nIters)
        local input, label, indices = unpack(sample)
        
        -- visualize
        --[[image.save('1.jpg', input[1])
        for j=1,16 do
            image.save('2.jpg', torch.mul(input[1][1],0.5):add(image.scale(label[1][1][j],256)))
            io.read()
        end]]

        local sInput = image.scale(input:view(opt[set .. 'Batch']*3, opt.inputRes,opt.inputRes), opt.outputRes)
                                        :view(opt[set .. 'Batch'], 3, opt.outputRes, opt.outputRes)
        local input_real, input_fake
        
        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
            sInput = applyFn(function (x) return x:cuda() end, sInput)
        end       

        if tag == 'train' then
            criterionD_fake = nn.ParallelCriterion()
            criterionD_fake:add(nn[opt.crit .. 'Criterion'](), -k_t)
            criterionD_fake:cuda() 
            
            model:zeroGradParameters()
            netD:zeroGradParameters()
            
            input_real = torch.cat(sInput, label[#label], 2)
            
            -- Discriminator          
            output_real = netD:forward(input_real):clone()
            errD_real = criterionD_real:forward(output_real, label[#label])
            df_D_real = criterionD_real:backward(output_real, label[#label])
            netD:backward(input_real, df_D_real)
            
            -- Generator
            output = model:forward(input)
            if opt.nStack == 1 then output = {output} end
            err_G1 = criterion:forward(output, label)
            df_G1 = criterion:backward(output, label)
            
            input_fake = torch.cat(sInput, output[#output], 2)
            
            -- update D: loss = err_real - k_t * err_fake, accumulate gradient
            output_fake = netD:forward(input_fake)
            errD_fake = criterionD_fake:forward({output_fake}, {output[#output]})
            df_D_fake = criterionD_fake:backward({output_fake}, {output[#output]})
            netD:backward(input_fake, df_D_fake[1])
            
            errD = errD_real + errD_fake
            optim.adam(evalFnD, paramD, optimStateD)
            
            -- update G: loss = err_G1 + opt.lambda_G * err_G2, accumulate gradient            
            err_G2 = criterionG2:forward({output_fake}, {output[#output]})
            df_G2_out = criterionG2:backward({output_fake}, {output[#output]})
            df_G2_in = netD:updateGradInput(input_fake, df_G2_out[1])
            df_G1[#df_G1] = df_G1[#df_G1] + df_G2_in[{{}, {4, 4+dataset.nJoints-1}}]
            if opt.nStack == 1 then df_G1 = df_G1[1] end
            model:backward(input, df_G1)
            
            errG = err_G1 + err_G2 
            optfn(evalFn, param, optimState)
            
            -- avg for log
            avgG1 = avgG1 + err_G1 / nIters
            avgG2 = avgG2 + err_G2 / nIters
            avgD_real = avgD_real + errD_real / nIters
            avgD_fake = avgD_fake + errD_fake / nIters
            
            -- update k_t
            local balance = opt.gamma * errD_real - err_G2 / opt.lambda_G
            k_t = k_t + opt.lambda_k * balance
            k_t = math.max(math.min(1, k_t), 0)
            
            measure = errD_real + math.abs(balance)
            
        else
            -- Do a forward pass and calculate loss
            output = model:forward(input)
            err = criterion:forward(output, label)
            avgG1 = avgG1 + err / nIters
            
            -- Validation: Get flipped output 
            output = applyFn(function (x) return x:clone() end, output)
            local flippedOut = model:forward(flip(input))
            flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

            -- Save sample
            local bs = opt[set .. 'Batch']
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))    
            if opt.nStack == 1 then output = {output} end
        end
        
        -- Calculate accuracy
        avgAcc = avgAcc + accuracy(output, label) / nIters
    end

    -- Print and log some useful metrics
    if tag == 'train' then
        print(string.format("      %s : Acc: %.4f avgG1: %.4f avgG2: %.4f avgD_real: %.4f avgD_fake: %.4f measure: %.4f k_t: %.4f"  
                                    % {set, avgAcc, avgG1, avgG2, avgD_real, avgD_fake, measure, k_t}))
        if ref.log[set] then
            table.insert(opt.acc[set], avgAcc)
            ref.log[set]:add{
                ['epoch     '] = string.format("%d" % epoch),
                ['loss      '] = string.format("%.6f" % avgLoss),
                ['acc       '] = string.format("%.4f" % avgAcc),
                ['LR        '] = string.format("%g" % optimState.learningRate)
            }
        end
    else
        print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgG1, avgAcc}))
    end

    if (epoch==1 or (opt.snapshot ~= 0 and epoch % opt.snapshot == 0)) and tag == 'train' then
        -- Take a snapshot
        model:clearState()
        netD:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'optimStateD.t7'), optimStateD)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
	    torch.save(paths.concat(opt.save, 'netD.t7'), netD)
    end
    
    if tag == 'predict' then
        predFilename = 'final_preds.h5'
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()    
    end
        
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
