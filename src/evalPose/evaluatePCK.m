
predidxs=16;
evalMode='PC';
bSave=0;
% implementation of PCK measure,
% as defined in [Sapp&Taskar, CVPR'13].
% torso height: ||left_shoulder - right hip||

% fprintf('evaluatePCK\n');

% if (nargin < 4)
%     bSave = true;
% end

range = 0:0.01:0.2;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end

% load ground truth
assert(strcmp(evalMode,'PC') || strcmp(evalMode,'OC'));
load(['joints-PC.mat']);
tableTex = cell(length(predidxs)+1,1);
% fprintf('evalMode: %s\n',evalMode);

pckAll = zeros(length(range),15,length(predidxs));

for i = 1:length(predidxs);
    
    % load predictions
    p = getExpParams(predidxs(i));
    %load(p.predFilename,'pred');
    ests = permute(prediction_all,[3,1,2]);
    pred = permute(ests,[3,2,1]);
    
    
    % compute distance to ground truth joints
    dist = getDistPCK(pred,joints(1:2,:,1001:2000));

    % compute PCK
    pck = computePCK(dist,range);
    
    % plot results
    [row, header] = genTablePCK(pck(end,:),' ');
    tableTex{1} = header;
    tableTex{i+1} = row;
    
    pckAll(:,:,i) = pck;
    
    auc = area_under_curve(scale01(range),pck(:,end));
%     plot(range,pck(:,end),'color',p.colorName,'LineStyle','-','LineWidth',3);
    fprintf('%s, AUC: %1.1f\n',p.name,auc);
end

if (bSave)
    fid = fopen([tableDir '/pck-' evalMode '.tex'],'wt');assert(fid ~= -1);
    for i=1:length(tableTex),fprintf(fid,'%s\n',tableTex{i}); end; fclose(fid);
end

