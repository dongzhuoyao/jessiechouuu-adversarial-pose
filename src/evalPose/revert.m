load('D:\evalLSP\pred\MyFCN\ests.mat');

for i=1:1000
    for j=1:14
        ests(i,j,1) = ests(i,j,1)*crop(i,4)/224;
        ests(i,j,2) = ests(i,j,2)*crop(i,3)/224;
        ests(i,j,1) = ests(i,j,1)+crop(i,2);
        ests(i,j,2) = ests(i,j,2)+crop(i,1);
    end
end

save('D:\evalLSP\pred\MyFCN\ests.mat','ests');