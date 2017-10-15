function p = getExpParams(predidx)

switch predidx
    case 0
        p.name = 'Pishchulin et., CVPR''13';
        p.predFilename = '';
    case 1
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_keypoints_lsp_oc';
        p.colorIdxs = [1 1];
    case 2
        p.name = 'Tompson et al., NIPS''14';
        p.predFilename = './pred/tompson14nips/pred_keypoints_lsp_pc';
        p.colorIdxs = [2 1];
    case 3
        p.name = 'Chen&Yuille, NIPS''14';
        p.predFilename = './pred/chen14nips/pred_keypoints_lsp_oc';
        p.colorIdxs = [5 1];
    case 4
        p.name = 'Ramakrishna et al., ECCV''14';
        p.predFilename = './pred/ramakrishna14eccv/pred_keypoints_lsp_oc';
        p.colorIdxs = [4 1];
    case 5
        p.name = 'Ouyang et al., CVPR''14';
        p.predFilename = './pred/ouyang14cvpr/pred_keypoints_lsp_oc';
        p.colorIdxs = [7 1];
    case 6
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_keypoints_lsp_pc';
        p.colorIdxs = [1 1];
    case 7
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_sticks_lsp_oc';
        p.colorIdxs = [1 1];
    case 8
        p.name = 'Pishchulin et., ICCV''13';
        p.predFilename = './pred/pishchulin13iccv/pred_sticks_lsp_pc';
        p.colorIdxs = [1 1];
    case 9
        p.name = 'Tompson et al., NIPS''14';
        p.predFilename = './pred/tompson14nips/pred_sticks_lsp_pc';
        p.colorIdxs = [2 1];
    case 10
        p.name = 'Ramakrishna et al., ECCV''14';
        p.predFilename = './pred/ramakrishna14eccv/pred_sticks_lsp_oc';
        p.colorIdxs = [4 1];
    case 11
        p.name = 'Chen&Yuille, NIPS''14';
        p.predFilename = './pred/chen14nips/pred_sticks_lsp_oc';
        p.colorIdxs = [5 1];
    case 12
        p.name = 'Ouyang et al., CVPR''14';
        p.predFilename = './pred/ouyang14cvpr/pred_sticks_lsp_oc';
        p.colorIdxs = [7 1];
    case 13
        p.name = 'Pishchulin et al., CVPR''13';
        p.predFilename = './pred/pishchulin13cvpr/pred_sticks_lsp_oc';
        p.colorIdxs = [8 1];
    case 14
        p.name = 'Kiefel&Gehler, ECCV''14';
        p.predFilename = './pred/kiefel14eccv/pred_sticks_lsp_oc';
        p.colorIdxs = [8 1];
    case 15
        p.name = 'Kiefel&Gehler, ECCV''14';
        p.predFilename = './pred/kiefel14eccv/pred_keypoints_lsp_oc';
        p.colorIdxs = [8 1];     
    case 16
        p.name = 'MyFCN';
        p.predFilename = './pred/MyFCN/ests';
        p.colorIdxs = [1 1];
    case 17
        p.name = 'wei';
        p.predFilename = './pred/wei/LSP_prediction_model_LSP_6s';
        p.colorIdxs = [1 1];
    case 18
        p.name = 'MyFCN_MSE';
        p.predFilename = './pred/MyFCN/estsMse';
        p.colorIdxs = [1 1];
end

p.colorName = getColor(p.colorIdxs);
p.colorName = p.colorName ./ 255;

end