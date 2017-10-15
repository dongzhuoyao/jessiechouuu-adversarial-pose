a = hdf5read('prediction/final_preds.h5', 'preds');
idxs = hdf5read('prediction/final_preds.h5', 'idxs');

[~, I] = sort(idxs);

b = a(1:2, [1:6,11:16,9,10], I);

prediction_all = permute(b, [2, 1, 3]);
save('prediction/LSPformatResult.mat', 'prediction_all');

evaluatePCP()
evaluatePCK()

