cor =zeros(14,1000);
node = [2,5,8,11];
leave = [1,3,4,6,7,9,10,12,13,14];

cor(1,:) = matches(1,:);
cor(2,:) = logical(matches(1,:))|logical(matches(2,:));
cor(3,:) = matches(2,:);
cor(4,:) = matches(3,:);
cor(5,:) = logical(matches(3,:))|logical(matches(4,:));
cor(6,:) = matches(4,:);
cor(7,:) = matches(5,:);
cor(8,:) = logical(matches(5,:))|logical(matches(6,:));
cor(9,:) = matches(6,:);
cor(10,:) = matches(7,:);
cor(11,:) = logical(matches(7,:))|logical(matches(8,:));
cor(12,:) = matches(8,:);
cor(13,:) = matches(9,:);
cor(14,:) = matches(9,:);

