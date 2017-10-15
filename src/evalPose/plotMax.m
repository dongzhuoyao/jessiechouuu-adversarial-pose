load('cor.mat');
load('max_value_stg6.mat');
load('joints-PC.mat');

mode=3;

if mode==1
    fail = find(cor==0);
    [f1,f2] = ind2sub([14,1000],fail);

    suc = find(cor==1);
    [s1,s2] = ind2sub([14,1000],suc);

    for i=1:size(fail,1)
        max_value_fail(i) = max_value(f2(i),f1(i));
    end

    for i=1:size(suc,1)
        max_value_suc(i) = max_value(s2(i),s1(i));
    end

    edges = 0:0.1:1;
    N = histcounts(max_value_fail,edges);
    plot(N,'color','blue');
    figure;
    M=histcounts(max_value_suc,edges);
    plot(M,'color','red');
elseif mode==2
    test = joints(:,:,1001:2000);
    occ = find(test(3,:,:)==1);
    [o1,o2] = ind2sub([14,1000],occ);
    for i=1:1054
    max_value_occ(i) = max_value(o2(i),o1(i));
    end
    vis = find(test(3,:,:)==0);
    [v1,v2] = ind2sub([14,1000],vis);
    for i=1:12946
    max_value_vis(i) = max_value(v2(i),v1(i));
    end
    
    edges = 0:0.1:1;
    N = histcounts(max_value_occ,edges);
    plot(0.1:0.1:1,N,'color','blue');
    figure;
    M=histcounts(max_value_vis,edges);
    plot(0.1:0.1:1,M,'color','red');
elseif mode==3
    test = joints(:,:,1001:2000);
    occ = uint8(find(test(3,:,:)==1));
    fail = uint8(find(cor==0));
    onf = intersect(occ,fail);
    [o1,o2] = ind2sub([14,1000],onf);
    for i=1:size(onf,1)
    max_value_onf(i) = max_value(o2(i),o1(i));
    end
    
    edges = 0:0.1:1;
    N = histcounts(max_value_onf,edges);
    plot(0.1:0.1:1,N,'color','blue');
end