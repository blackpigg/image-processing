epoch = 1;
result= cell(0);
result2 = zeros(10,10000);
result3 = zeros(10,10000);

for i = 1:2000
    result{end+1} = squeeze(responseBackup{i}(14).x);
end

for i = (epoch-1)*100+1:epoch*100
    result2(:,((i)-1)*100+1:100*(i)) = result{i};
end

shuffle2 = shuffle((epoch-1)*10000+1:epoch*10000);
result3 = result2(:,1:10000);