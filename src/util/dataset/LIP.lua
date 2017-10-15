local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.nJoints = 16
    self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
    self.flipRef = {{1,6},   {2,5},   {3,4},
                    {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},    {4,5,2},
                        {5,6,2},    {9,10,0},   {13,9,3},   
                        {11,12,3},  {12,13,3},  {14,9,4},
                        {14,15,4},  {15,16,4}}

    local annot = {}
    local tags = {'imgname','part','center','scale','visible','istrain'}
    local a = hdf5.open('../data/LIP/lip.h5'),'r')
    for _,tag in ipairs(tags) do 
      annot[tag] = a:read(tag):all() 
    end
    a:close()
    annot.center:add(1)

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.index:size(1))
        opt.idxRef = {}
        opt.idxRef.valid = allIdxs[annot.istrain:eq(2)]
        opt.idxRef.train = allIdxs--[annot.istrain:eq(1)]

        -- Set up training/validation split
        --local perm = torch.randperm(opt.idxRef.train:size(1)):long()
        --opt.idxRef.valid = opt.idxRef.train:index(1, perm:sub(1,opt.nValidImgs))
        --opt.idxRef.train = opt.idxRef.train:index(1, perm:sub(opt.nValidImgs+1,-1))

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     --valid=opt.idxRef.valid:numel(),
                     valid=opt.idxRef.valid:numel()
                     }

    -- For final predictions
    opt.validIters = self.nsamples.valid
    opt.validBatch = 1
    
    print(self.nsamples)
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    local filename = ffi.string(self.annot.imgname[idx]:char():data())
    filename = string.split(filename,'*')[1]
    if self.annot.istrain[idx]==1 then
        return paths.concat(projectDir .. '/data','LIP','train_images',filename)
    else
        return paths.concat(projectDir .. '/data','LIP','val_images',filename)
    end
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    -- Small adjustment so cropping is less likely to take feet out
    s = s * 1.8
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset


