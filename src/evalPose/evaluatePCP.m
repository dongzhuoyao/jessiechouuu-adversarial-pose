% implementation of strict PCP measure,
% as defined in [Ferrari et al., CVPR'08]

name = 'CPM';

bSave = true;
range = 0.5;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end
load('joints-PC.mat','joints');

tableTex = cell(1+1,1);


% load predictions
%load_path = '/tmp/LSPformatresult.mat';
%load(load_path);
ests = permute(prediction_all,[3,1,2]);
pred = permute(ests,[3,2,1]);
%disp(load_path)

% compute distance to ground truth joints
dist = getDistPCP(pred,joints(1:2,:,1001:2000));

% compute PCK
pcp = computePCP(dist,range);

% plot results
[row, header] = genTablePCP(pcp(end,:),name);
tableTex{1} = header;
%tableTex{i+1} = row;
