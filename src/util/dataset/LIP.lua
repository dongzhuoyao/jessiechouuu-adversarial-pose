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
    local a = hdf5.open('../data/LIP/annotations/lip.h5','r')
    for _,tag in ipairs(tags) do 
      annot[tag] = a:read(tag):all() 
    end
    a:close()

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1,annot.istrain:size(1))
        opt.idxRef = {}
        opt.idxRef.train = allIdxs[annot.istrain:eq(1)]
        opt.idxRef.valid = allIdxs[annot.istrain:eq(2)]
        opt.idxRef.test = allIdxs[annot.istrain:eq(0)]

        torch.save(opt.save .. '/options.t7', opt)
    end

    self.annot = annot
    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}
    
    print(self.nsamples)

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    local filename = ffi.string(self.annot.imgname[idx]:char():data())
    filename = string.split(filename,'*')[1]
    return paths.concat(projectDir .. '/data','LIP',filename)
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx)
    local pts = self.annot.part[idx]:clone()
    local c = self.annot.center[idx]:clone()
    local s = self.annot.scale[idx]
    s = s * 1.8
    return pts, c, s
end

function Dataset:normalize(idx)
    return self.annot.normalize[idx]
end

return M.Dataset


